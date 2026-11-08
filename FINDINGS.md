# WebGPU cluster research findings — 2026-11-08

Data source: user-run browser-local measurements on the advanced WebGPU benchmark, shared in chat on 2026-11-08. Reported metrics use five measured trials per mode, after two warmups, unless noted. These are observed measurements, not a controlled hardware-independent performance characterization.

## GPU residency (65,536 f32 elements, 16 dispatches)

| Mode | Median end-to-end duration | Mean | Validation |
| --- | ---: | ---: | --- |
| GPU-resident, separate submissions | 4.2 ms | 4.14 ms | 5/5 |
| Readback after every dispatch | 53.2 ms | 54.28 ms | 5/5 |

The readback-each path is 12.67x slower by median elapsed time. It reads 4 MiB of intermediate results (16 x 256 KiB) versus zero intermediate bytes for the resident path. Both read a final 256 KiB result. This includes browser overhead, mapping and synchronization and is **not** isolated PCIe transfer time.

Earlier same-device checkpoint experiment (four 64 KiB chunks): median snapshot 10.8 ms, restore 0.2 ms, and verification 3.9 ms; all five trials validated. Restore is to a second GPU buffer on the same device, not a distinct node.

## Dispatch batching (65,536 elements, 16 dispatches)

| Mode | Median | Mean | Min | Max | Valid |
| --- | ---: | ---: | ---: | ---: | --- |
| One batched command buffer | 3.6 ms | 3.52 ms | 2.9 ms | 3.9 ms | 5/5 |
| Separate submissions, no intermediate readback | 4.2 ms | 4.26 ms | 3.3 ms | 5.6 ms | 5/5 |
| Readback after each dispatch | 63.3 ms | 62.24 ms | 57.2 ms | 66.3 ms | 5/5 |

At these settings, batching improved median latency by (4.2-3.6)/4.2 = 14.3% compared to separate submissions (1.17x). Readback after each dispatch was 63.3/3.6 = 17.58x slower than batching. In the readback-each path, intermediate readback time had median 59.4 ms of 63.3 ms total; stage/map/scheduling are included. Five trials are insufficient to claim a stable 0.6 ms gain in every environment.

## Sequential checkpoint chunk scaling (constant 256 KiB state)

| Chunks | Chunk size | Median snapshot | Mean snapshot | Median restore | Correct |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 256 KiB | 3.7 ms | 3.60 ms | 0.3 ms | 5/5 |
| 4 | 64 KiB | 8.9 ms | 9.38 ms | 0.3 ms | 5/5 |
| 16 | 16 KiB | 43.4 ms | 40.54 ms | 0.2 ms | 5/5 |
| 64 | 4 KiB | 167.3 ms | 166.18 ms | 0.3 ms | 5/5 |

Sixty-four sequential readbacks took 167.3/3.7 = 45.22x longer than one readback of identical total size, on these runs. The existing sequential implementation allocates a staging buffer, submits a copy, waits for mapAsync, copies to JS memory, and cleans up **once per chunk**. These results suggest that *repeated synchronization points*, not merely splitting data, are the main source of the penalty.

The observed sequential capture times are **not** proof that a 4 KiB transport chunk is intrinsically slow. The restoration path queues writes and synchronizes once, which helps explain why median restore time remained near 0.2–0.3 ms across all chunk counts. Its small duration makes it prone to clock granularity and scheduling noise.

## New test: capture strategy comparison

Advanced page: /advanced.html → **Run capture strategy comparison**.

For 1, 4, 16 and 64 chunks, compare identical buffer contents and exact post-restore output across:

1. **Sequential:** each chunk separately copied from GPU, mapped, then CPU-copied, with its own submit/map.
2. **Batched copies:** prepare one staging buffer per chunk, encode all GPU copies in a single command buffer, submit once; then await all mapAsync calls and copy their bytes.
3. **Single GPU readback + CPU-side split:** read the full buffer once, split CPU-owned bytes into the same number of chunks using Uint8Array slices.

Metrics: end-to-end snapshotMs, restoreMs, verificationMs, cpuSplitMs, gpuSubmissions, gpuMapCalls, snapshotMiBPerSec and element-level validation. Trials alternate/reverse configuration order to reduce order bias.

**Hypothesis, not established finding:** batched copies or CPU-side chunking should remove most of the 64-chunk sequential readback penalty. New run data is needed to confirm this.

## Architectural implications and limitations

- Keep dependent intermediate data on the GPU where practical and batch known dispatches.
- Choose synchronization/checkpoint boundaries intentionally; avoid mandatory GPU→CPU waits after every dispatch.
- Prefer extracting GPU state efficiently before transport-oriented chunking, **subject to new experiment results**.
- Buffer snapshots alone do **not** preserve running shader execution, pipeline bindings, network state, or arbitrary application logic. Manual import/export demonstrates buffer reconstruction only.
- WebGPU timing includes JS event-loop, GPU queue and browser effects. Small samples and non-isolated workloads introduce noise.
- No measurements yet confirm actual network throughput, cross-browser transfer latency, GPU device loss recovery, or cross-device live migration.

## Next measurement sequence

1. Run the capture strategy comparison at defaults (65,536 elements, 5 trials, 2 warmups). Export JSON.
2. Run larger state scaling to see when bytes, not synchronization, dominate.
3. Run checkpoint frequency experiment to quantify cost against uninterrupted compute.
4. Export a portable checkpoint and restore on another browser/device; report both adapters and validation.
5. Later: implement a network relay and progress-bearing resumable tasks; measure complete migration time and continued output correctness.
