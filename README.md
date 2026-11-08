---
title: Cluster WebGPU Lab
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Cluster WebGPU Lab

A small, independent browser-side research lab inspired by [wgpuclustertest](https://github.com/Wuhpondiscord/wgpuclustertest).

Additional experiments: [GPU residency and checkpoint testing](./residency.html).\n\nExperiments: adapter capabilities, bounded vector addition (same 64-thread workgroup shape as the cluster), GPU readback, optional GPU timestamp queries, and instrumented WGSL execution counters. No accounts, coordinator, P2P network, or ML model downloads.

The Hugging Face Docker Space serves static files; all GPU work runs locally in the visiting browser. Results stay local unless manually exported. Requires WebGPU and a secure context.

## Run locally

```bash
python3 -m http.server 7860
```

Open http://localhost:7860. Docker: `docker build -t cwgpu-lab . && docker run -p 7860:7860 cwgpu-lab`.

## Notes

- Wall-clock times include asynchronous GPU completion/readback, not just shader execution.
- Timestamp-query GPU time is only shown when supported and successfully measured.
- Allocated bytes are requested application buffer sizes, not actual VRAM consumption.
- Instrumented shader atomics add overhead and measure logical invocation counts, not hardware instructions.
- Each test is capped at 500,000 float elements, matching the main cluster's bounded vector-add path.
- GPU work is opt-in via the Run button; no hidden background computation.
