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
