# ring: scientific and implementation observation

This is a real two-device ring, separate from the single-V100 performance population. Both buffers are checked in full after every trial. All four versions call the same shared CUDART cudaMemcpyDefault function and fence both devices at the end of the measured region; no peer-enable calls are issued. Framework versions differ in allocation API, not translated kernels. Original timing without an explicit completion guarantee and the prior one-device self-copy diagnostic are not completed-transfer measurements.

## Reference audit

Existing CUDA.jl copyto! may automatically enable peer access and uses an asynchronous driver path. Study retains framework allocation but calls the same shared CUDART cudaMemcpyDefault API as native, with explicit both-device completion fences. Original native timer lacks those fences.

## Numerical and performance evidence

Configuration `ring_gpu5_gpu6_n65536_round10_fenced_legacy_alloc_v2`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T165903935088Z_ring/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

First CUDA.jl/KA host programs fail with CUDART invalid-value when using default stream-ordered pooled allocations. A guarded four-element diagnostic succeeds with JULIA_CUDA_MEMORY_POOL=none; allocation policy changes outside timing. Device-kernel count remains0; this is a runtime interoperation repair, not a framework feature unsupported claim. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/ring-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
