# scatter: scientific and implementation observation

All twenty native type/operator cases are retained. The named ±1 fixture makes multiplication and division order-independent and permits exact full-output checks. Julia uses a uniform raw-bit CAS implementation, an explicit agent algorithm choice rather than evidence that native atomic operations are unavailable. Comparisons therefore include this mapping change and cannot isolate abstraction overhead.

## Reference audit

Historical CUDA.jl scatter_touch! merely increments output slots and prints PASS for all twenty labels; it neither reads source indices nor computes reductions. Study reference implements genuine contended CAS reductions.

## Numerical and performance evidence

Configuration `scatter_all20_n4096_collision2_signedunit_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T163607327472Z_scatter/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Device kernel candidate1 compiled/warmed all twenty specializations. Shared host fixture required two repairs: B2 comprehension spacing, then R1 host reset copy from ReinterpretArray falling back to forbidden GPU scalar indexing. Materialize the host reinterpret buffer with collect; never enable scalar GPU fallback. Three process checks each for CUDA.jl/KA; JACC inherits fixed fixture. Historical CUDA.jl attempts unknown. Reference S1 semantic repair is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/scatter-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
