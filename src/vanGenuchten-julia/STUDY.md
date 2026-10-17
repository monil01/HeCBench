# vanGenuchten: scientific and implementation observation

This hydrology workload evaluates nonlinear constitutive physics independently per cell. It exercises FP64 powers, square root and saturation branching; its map motif alone does not establish a compute bottleneck without counters. The named fixture uses native input initialization, covers unsaturated and saturated pressure, and compares all three scientific fields. All framework device kernels use native-equivalent256-thread geometry and no shared memory. Native tolerance1e-3 is retained, with stricter finite-value checks.

## Reference audit

Historical CUDA.jl smoke_kernel writes one integer marker then prints hardcoded PASS; no van Genuchten constitutive equations or scientific outputs. Replaced only study entrypoint with genuine per-cell FP64 equations.

## Numerical and performance evidence

Configuration `soil_fp64_16x16x16_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T160605690412Z_vanGenuchten/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/vanGenuchten-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
