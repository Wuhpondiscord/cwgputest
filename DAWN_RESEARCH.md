# Dawn internals: research notes and regression plan

Date: 2026-11-08. Source: selected files supplied by the researcher from `google/dawn`, with supporting inspection of upstream `RingBufferAllocator.h/.cpp` and `ResidencyManagerD3D12.cpp`. These are **source-review candidates**, not proven security vulnerabilities. No native fault-injection tests have run.

## Research scope and evidence

Paths examined:
- `src/dawn/native/DynamicUploader.cpp` (4 MiB staging ring, 16 MiB pending-submit threshold, serial reclamation)
- `src/dawn/native/Queue.cpp` (queue serials, callbacks, texture-copy helper)
- `src/dawn/native/Buffer.cpp` (mapping state, mapping bounds and usage validation)
- `src/dawn/wire/client/Buffer.cpp` (client mapping, serialization and shared-memory data)
- `src/dawn/native/d3d12/QueueD3D12.cpp` (D3D12 command-list and fence serial handling)
- `src/dawn/native/d3d12/BufferD3D12.cpp` (resource heap selection, buffer mapping and residency)
- Cross-checked `src/dawn/native/RingBufferAllocator.h/.cpp` and `src/dawn/native/d3d12/ResidencyManagerD3D12.cpp`.

## Security candidates

### Candidate A — mapping-error residency-lock balance

In the D3D12 buffer mapping path, a successful `ResidencyManager::LockAllocation` is followed by `ID3D12Resource::Map`. The latter can return a failing HRESULT. A source-level concern is whether an early error return leaves a residency lock held. The residency manager increments an internal lock counter and only returns a heap to the eligible-for-eviction tracking list when unlocked.

**Status:** plausible lifecycle/error-unwind defect; not confirmed. Need trace `Buffer::MapInternal`, its callers and device-loss cleanup in complete native source, and demonstrate a controlled mapping failure with a mock/fault-injected D3D12 resource. Check lock count and heap residency after failed mapping. A failure might coincide with device loss and have no persistent effect. Do not report as an exploitable denial of service without demonstrating sustained resource retention.

### Candidate B — ignored D3D12 fence-event HRESULT

`QueueD3D12.cpp` delegates to `ID3D12Fence::SetEventOnCompletion` without checking its HRESULT. **Confirmed code observation**, but the runtime consequence is unverified. Inspect all callers, event-handle lifetime, and how a failing event-registration operation propagates to waiters. Native fault-injection can test error handling without intentional hangs. A browser-visible hang is not established.

### Candidate C — `Align(allocationSize,8)` precondition

`DynamicUploader::Reserve` aligns a `uint64_t` before checking a locally explicit maximum. `dawn::Align` assumes arithmetic will not overflow; calling this on an unconstrained near-`UINT64_MAX` size would violate its precondition. **Not a demonstrated overflow vulnerability:** normal upstream validation and API bounds may prohibit such values. Next: enumerate and test callers using bounded native unit tests / checked arithmetic. Do not submit huge browser-side allocations.

### Candidate D — texture stride underflow / overflow preconditions

`Queue.cpp::CopyTextureData` computes `srcBytesPerRow * (srcRowsPerImage - dstRowsPerImage)` as unsigned arithmetic. Safety depends on `ValidateWriteTexture` and layout normalization guaranteeing the expected row relationship, and on checked spans. **No confirmed out-of-bounds access.** Next: trace `ValidateWriteTexture` across single-layer, multilayer and compressed-format cases; add native validation regression cases for zero sizes and near-boundary row layouts without bypassing validation.

### Candidate E — new ring fallback alignment

New `RingBufferAllocator` fallback call omits the `offsetAlignment` argument; default is 1, but the new empty ring begins at offset 0. Zero is aligned for all valid positive power-of-two alignments. **Not presently a bug.** A possible code-hygiene improvement is passing the alignment explicitly to preserve intent.

## Cluster-related mechanisms

- Smaller uploads are suballocated in 4 MiB staging rings, associated with pending execution serials; larger uploads use dedicated staging buffers. **This is upload-side policy, not a 4 MiB readback or storage limit.**
- Pending upload memory at 16 MiB may trigger submission, to avoid indefinitely retaining resources awaiting command completion; it does not cap total GPU memory.
- Ring allocations are reclaimed based on completed GPU execution serials, a useful design for pooled checkpoint staging buffers.
- D3D12 has readback/upload/default heap placement distinctions and explicit residency management.
- D3D12 queue handles command list recycling and GPU completion fencing. For reliable checkpointing, the scheduler should identify completed execution boundaries, not assume submission equals completion.
- Browser wire/client mapping has an additional data-serialization/copy path; WebGPU JavaScript timing includes more than native device transfer.

## Browser regression page

Open `/dawn.html` in the existing Space. This is **not native Dawn fault injection**. It exercises bounded valid and invalid WebGPU API behavior only.

| Probe | What it can establish | What it cannot |
| --- | --- | --- |
| Copy beyond source bounds | WebGPU copy validation is enforced | Native pointer safety for arbitrary wire input |
| Misaligned copy offset | Alignment rejection is enforced | Internal allocator alignment correctness |
| Out-of-range and misaligned `mapAsync` | Browser mapping validation | D3D12 mapping-failure residency lock balance |
| Mapping already-mapped buffer | Rejection of invalid mapping lifecycle | Race safety for every native callback ordering |
| Destroy while mapping pending | Rejection/abort is handled | Absence of memory corruption |
| Repeated staging map/unmap | Small mapping cycle preserves bytes | Large GPU resource residency lifetime |
| 12 sequential vs 12 batched readbacks | GPU synchronization/copy overhead at 4 KiB | Exact GPU transfer bandwidth or native allocation cost |

All probes are opt-in, local, capped to 16 KiB per buffer, and exported as JSON including errors and timings. An **unexpected acceptance** is a triage lead, not a confirmed security vulnerability. A **pass** means the browser behaved according to the test's limited expectation; it cannot rule out memory-safety bugs. Do not intentionally exhaust memory or crash the browser.

## Next investigations

1. Run browser probes on the target Chrome + D3D12 system and export JSON, then repeat on another WebGPU backend if available.
2. Review native `Buffer::MapInternal` error/unwind call graph and add a controlled D3D12 failure-injection unit test for lock balancing. Native C++ tests cannot run in a static Hugging Face Space.
3. Audit fence event registration call sites and error handling, using native mock failure injection rather than a real indefinite wait.
4. Audit `ValidateWriteTexture` invariants and test boundary layouts under Dawn's native tests.
5. Add fine-grained tracing in a locally built Dawn to separate D3D12 fence latency, mapping, wire transport and browser callback delay; compare with the existing 308-run browser benchmark.

## Research integrity

No code changes were made to upstream `google/dawn`. The research lab is browser JavaScript served as static files; it cannot access private native pointers, D3D12 HRESULTs, allocator counters or GPU driver internals from ordinary WebGPU. These native claims remain open hypotheses until specifically tested.


## First browser regression result (2026-11-08, Chrome 157, NVIDIA Lovelace)

User-provided run: `2026-11-08T17:12:30.617Z` through `17:12:30.737Z`; browser `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36`; adapter reports vendor `nvidia`, architecture `lovelace`. The WebGPU adapter metadata does **not** identify which native backend Chrome selected; do not assume D3D12 from these fields alone.

**9/9 bounded browser-facing probes reported pass; 0 unexpected acceptances; 0 inconclusive.**

| Probe | Outcome | Reported elapsed |
| --- | --- | ---: |
| Copy beyond source-buffer bounds | Correctly rejected | 1.6 ms |
| Non-4-byte-aligned buffer copy | Correctly rejected | 0.7 ms |
| Out-of-range `mapAsync` | Correctly rejected | 0.8 ms |
| Misaligned `mapAsync` offset (4 bytes, required 8) | Correctly rejected | 0.2 ms |
| Second `mapAsync` on already mapped buffer | Correctly rejected | 0.3 ms |
| Destroy during pending `mapAsync` | Aborted with `AbortError` | 0.1 ms |
| Reuse 16-KiB staging buffer for 3 roundtrips | Exact bytes validated | 9.7 ms |
| 12 sequential 4-KiB readbacks (12 submits + serial maps) | Exact bytes validated | 31.3 ms |
| 12 batched 4-KiB readbacks (1 submit + concurrent maps) | Exact bytes validated | 2.9 ms |

One-run batched speedup: `31.3 / 2.9 = 10.79x` (90.7% shorter elapsed). Both paths read 48 KiB in total; synchronization schedule differs. **Do not equate this difference to pure PCIe transfer speed or Dawn-only overhead.** Measurements are single-pass and order-dependent, so next experiment should repeat warmups/trials and alternate the order of the two modes.

Expected validation and map cancellation **do not test the native D3D12 failure-injection hypotheses**. The mapping residency-lock and unchecked fence HRESULT candidates remain open; no evidence of memory corruption or exploitable vulnerability was produced. The device's native backend and driver version were not recorded.

### Follow-up priorities informed by these results

1. Add 5–20-trial alternating-order performance comparison with summaries and bounded work, plus browser/device metadata and failure counts. Compare 1, 4, 12 and 32 buffers to separate per-map and submission overhead.
2. Add safe mapping lifecycle transitions: `unmap` after successful read, `destroy` before/after completed maps, and expected rejection for duplicate pending mapping, with clean isolated devices on errors.
3. For native security hypotheses, use a local Dawn native build with injected D3D12 mapping failure and instrumented lock balancing; then injected fence event registration failure. Browser JavaScript cannot verify their internal invariants.
4. Audit native write-texture validation and alignment preconditions with focused unit tests and fuzzing. No oversize or malformed-wire stress testing belongs in the public Space.


## Follow-up suite implementation (2026-11-08, awaiting browser measurements)

A second independently exportable suite has been added to `/dawn.html`. Press **Run follow-up tests**, then export `dawn-followup-*.json`.

### Performance

Run 1, 4, 12 and 32 readback buffers of 4 KiB each, using sequential mapping/submission versus one batched submission and concurrent mapping. Default 2 warmup rounds and 5 measured rounds. Configuration accepts 1–20 measured trials and 0–5 warmups. Reverse size order on alternating rounds; alternate method order to reduce first-test bias. Verify all 1024 `u32` values in **every** buffer, not just the first element. Capture individual wall-clock samples, number of submissions and map calls, and aggregate medians, means, min/max and p95 plus median sequential/batched speedup. The maximum data copied per configuration is 128 KiB; the largest individual GPU buffer is 4 KiB. Each configuration includes resource allocation and cleanup, so timings are application end-to-end, not pure GPU-only.

**Caveat:** the round alternation is deterministic and not a fully randomized or temperature-controlled methodology. The tests use multiple buffers to study mapping synchronization; they do not expose Dawn internal tracing.

### Security / lifecycle

Five new bounded cases:
- `unmap()` while `mapAsync()` is pending should abort the pending mapping.
- A second `mapAsync()` while the first is pending should be rejected while the first one still completes normally.
- A successful `mapAsync()`, `unmap()`, and second mapping should validate exact returned data.
- `destroy()` after a successful mapping should prevent subsequent mapped-range retrieval.
- A `mapAsync()` on an already destroyed buffer should reject.

All record `pass`, `unexpected-acceptance`, or `inconclusive` with observed errors and elapsed time. A passing test is evidence of limited API-level rejection or lifecycle behavior only. **Neither this suite nor its original predecessor can confirm the suspected native D3D12 residency-lock leak or ignored fence HRESULT.**

### Still required for native candidates

A local Windows/Dawn D3D12 build with controllable `ID3D12Resource::Map` and `ID3D12Fence::SetEventOnCompletion` failure is required to determine whether the suspected error-handling defects are real and observable. Native test coverage should inspect resource-lock counters and bounded waiter error propagation respectively. No native fault injection has been executed or claimed.


## Second browser regression run — extended performance and lifecycle (2026-11-08)

Source: user-exported `dawn-followup` version 2; started `2026-11-08T17:17:52.068Z`, finished `17:17:53.190Z`. Browser: Chrome 157 on Windows 10/11 user-agent platform `Windows NT 10.0`; adapter vendor `nvidia`, architecture `lovelace`, native graphics backend **not independently identified**. Configuration: 2 warmup rounds; 5 measured rounds per case; 4 KiB staging buffers; 1/4/12/32 buffer counts; sequential or batched; alternating mode and size ordering.

**40/40 measured performance cases validated, zero recorded failures. Five of five extended mapping-lifecycle checks passed, zero unexpected acceptances, zero inconclusive.**

### Median end-to-end readback latency

| 4 KiB buffers | Total bytes | Sequential submits | Batched submits | Sequential median | Batched median | Ratio, sequential/batched |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 KiB | 1 | 1 | 2.7 ms | 3.6 ms | 0.75x |
| 4 | 16 KiB | 4 | 1 | 14.2 ms | 2.8 ms | 5.07x |
| 12 | 48 KiB | 12 | 1 | 32.0 ms | 3.1 ms | 10.32x |
| 32 | 128 KiB | 32 | 1 | 80.3 ms | 4.1 ms | 19.59x |

Measured `p95` (5 samples; effectively the observed maximum):
- 1 buffer: sequential 3.4 ms / batched 4.1 ms.
- 4 buffers: sequential 14.8 ms / batched 3.1 ms.
- 12 buffers: sequential 42.4 ms / batched 4.0 ms.
- 32 buffers: sequential 92.9 ms / batched 4.4 ms.

At 32 buffers, reduction vs. sequential median is `(80.3-4.1)/80.3 ≈ 94.9%`. At 1 buffer, methods have the same count of submissions and mapping calls; small timing differences should not be interpreted as fundamental performance differences. The general trend reflects serial synchronization overhead (GPU submission + wait/map for each buffer); the experiment does **not** isolate Dawn-only, wire-only, or hardware DMA times. Five trials and ordered execution remain insufficient to define universal cross-device performance.

### Lifecycle observations (5/5 passed)

| Case | Browser-observed result |
| --- | --- |
| `unmap()` with pending `mapAsync()` | Promise rejected with `AbortError` |
| Duplicate pending `mapAsync()` | Second request rejected with `OperationError`; first completed |
| Completed map → unmap → remap | Two exact 16-byte read cycles passed |
| Destroy after completed mapping | `getMappedRange()` rejected with `OperationError` |
| `mapAsync()` after destruction | Rejected with `OperationError` and validation error |

There were **no reported browser API validation anomalies**, but no assertion about D3D12 residency-lock balance or fence HRESULT can be drawn from these cases. Their native error paths remain open hypotheses, not confirmed vulnerabilities.

### Updated next steps

1. Benchmark GPU-resident multi-dispatch execution with staged periodic asynchronous checkpointing and triple-buffer staging pools, while keeping a bounded memory budget. Distinguish capture throughput from restore cost and end-to-end migration.
2. Run repeated benchmarks in a second browser/backend and record `chrome://gpu` backend details manually if available.
3. Move the D3D12 residency-lock and fence-HRESULT candidates into a native Dawn fault-injection harness; assert cleanup and non-hanging error signaling. Such tests cannot run from the static public browser lab.
4. Avoid claiming exploitable memory corruption, a native CVE, or absence of security bugs based on expected public WebGPU validation alone.


## Third-phase implementations: overlap test + D3D12 error-unwind model (2026-11-08)

### Bounded overlapped WebGPU checkpoint experiment

The existing `/dawn.html` now embeds `dawn-overlap.js`, providing a blocking vs. overlapped capture comparison. Default: 65,536 `u32` elements (256 KiB), 32 compute dispatches, 1 intermediate checkpoint every 4 dispatches (7 intermediate snapshots), 3 staging slots, 2 warmup rounds, 5 measured rounds. The shader adds 1 per element per dispatch; every checkpoint and the final readback are checked against expected GPU state.

- Blocking: submit a compute pass + checkpoint copy, then await that staging buffer's `mapAsync` and verify it before submitting the next dispatch.
- Overlapped: submit the same compute pass + copy, immediately request `mapAsync`, then enqueue later dispatches without awaiting that map. When all staging slots are in use, await and verify the oldest map, then reuse the slot. The pool has a strict configurable maximum of 1–4 staging buffers.
- Both have the **same GPU dispatch count, compute submissions, and snapshot bytes**. This isolates the benefit of avoiding CPU-side blocking after each snapshot. All submissions still share one WebGPU queue; this test does **not prove** copy/compute hardware concurrency. 
- The exported JSON contains trial order, exact snapshot validation, elapsed times, mapping wait time, copy-to-mapped latency, checkpoint count and bytes, in-flight map high-water mark and errors. Test size is bounded to 500,000 elements and 64 dispatches.
- On failure, partial JSON remains exportable. A browser/device test run is needed before any performance claim.

### Standalone C++ fault-unwind model

`native-dawn-tests/fault_harness.cpp` implements deterministic injected success/failure for **modeled** residency lock → D3D12 Map and D3D12 fence event registration paths. Its seven assertions exercise the original control-flow concern, RAII cleanup, successful map lifetime, lock failure, ignored fence failure and checked fence failure/success.

**Crucial distinction:** This is a *standalone native C++ test double*, not compiled against Dawn or Direct3D 12. Passing the model does not confirm a Dawn implementation defect, rule out one, or establish an exploitable native issue. Its purpose is to make the proposed correction's error-unwind contract executable on both Linux and Windows while a proper Dawn-specific injection seam is built.

`.github/workflows/dawn-fault-model.yml` compiles and executes the standalone model on Windows and Linux. See `native-dawn-tests/README.md` for manual compilation and the still-open work needed to instrument actual `BufferD3D12.cpp::MapInternal` and `QueueD3D12.cpp::SetEventOnCompletion`. A second ignored fence HRESULT occurs in `BufferD3D12.cpp::SynchronizeBufferBeforeMapping` and should also be audited.

**Pending:** actual Windows Dawn native fault-injection tests, real HRESULT/COM failure handling, and verified browser trial results. No upstream Dawn patches were made.


## Third browser follow-up (2026-11-08T17:25 UTC) and overlap compile fix

The user reran the 40-sample follow-up on Chrome 157 / NVIDIA Lovelace (5 trials, 2 warmups). **40/40 readback cases valid, all five mapping lifecycle tests passed, no failures**. Medians (sequential vs batched): 1 buffer 4.0/3.2 ms (1.25x); 4 buffers 12.3/3.3 ms (3.73x); 12 buffers 33.8/3.0 ms (11.27x); 32 buffers 88.7/3.2 ms (27.72x). The 32-buffer improvement remains robust across both runs, but outliers and browser scheduling affect per-run ratios.

The first attempt to run the new `dawn-checkpoint-overlap` experiment stopped at shader compilation with `Error: 'target' is a reserved keyword`. No overlap measurements were recorded. `dawn-overlap.js` was corrected to use WGSL identifier `outputState`; setup errors are now recorded in exported report `errors`. The corrected shader has **not yet been validated by a browser run**.
