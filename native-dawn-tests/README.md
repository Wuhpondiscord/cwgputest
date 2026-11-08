# D3D12 failure-injection research harness

This directory contains `fault_harness.cpp`: a **self-contained C++ control-flow test double**, runnable on Windows or Linux, and intended to help prepare and reason about Dawn-native regression tests. **It does not link against Dawn, call D3D12, inspect Dawn residency counters, inject real COM failures, or demonstrate an exploitable vulnerability.** A passing run validates only the failure-unwind design in the surrogate.

## Run the standalone design test

On Linux/macOS or Windows with GCC/Clang:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -pedantic fault_harness.cpp -o fault_harness
./fault_harness
```

On a Visual Studio Developer Command Prompt:

```bat
cl /nologo /std:c++17 /EHsc /W4 fault_harness.cpp /Fe:fault_harness.exe
fault_harness.exe
```

Seven assertions cover baseline missing cleanup, proposed RAII unlock on mapping error, normal map-unmap ownership, failure before acquiring lock, ignored fence error, checked fence error, and success.

## Native Dawn-specific regression integration still needed

1. Checkout the exact `google/dawn` revision and build its D3D12 native tests on Windows using the upstream build directions. A standalone WebGPU static site cannot load or instrument the browser's copy of Dawn.
2. Introduce a **test-only, scoped interception seam** just before the call to `GetD3D12Resource()->Map(...)` in `src/dawn/native/d3d12/BufferD3D12.cpp::MapInternal`. When activated in a unit test, inject a failing `HRESULT` and confirm `ResidencyManager::UnlockAllocation` runs exactly once after a previously successful `LockAllocation`. Use RAII/unwind cleanup within `MapInternal`; disarm it only after the underlying resource maps successfully. Verify normal `UnmapImpl` remains the sole owner of the lock on the success path. Restrict fault injection to the native test build; never expose it to web content.
3. In `src/dawn/native/d3d12/QueueD3D12.cpp::SetEventOnCompletion`, the current method returns `void` and ignores the HRESULT. Trace the complete virtual interface and its callers. A correction may require changing the base/overrides/callers to propagate a `MaybeError`; do not modify just the override. Inject a failing result in a native test and assert that callers detect it without blocking indefinitely.
4. Additionally audit `BufferD3D12.cpp::SynchronizeBufferBeforeMapping`, which directly calls `d3dFence->SetEventOnCompletion(..., nullptr)` without checking the HRESULT. This second call site is relevant to the same failure class.
5. Run Dawn native tests (including the D3D12 backend), sanitizer/debug configurations as supported, and Chromium integration tests before declaring a bug fixed.

## Why the source paths matter

`MapInternal` locks a residency-managed heap and then calls `ID3D12Resource::Map`. A failing `Map` could skip normal `UnmapImpl`, but the full native cleanup must be checked before assigning impact. In the fence path, ignoring `SetEventOnCompletion` HRESULT is observable, but whether it causes a user-visible hang requires inspecting caller behavior and native injected failures.

**This harness does not verify those Dawn-specific claims.** It intentionally avoids real GPU exhaustion, driver faults, or intentionally hanging the browser.
