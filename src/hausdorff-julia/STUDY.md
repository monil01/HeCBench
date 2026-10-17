# hausdorff: scientific and implementation observation

The native routine computes squared Hausdorff distance and returns two directional scalar reductions. The study confirms both complete outputs against exhaustive CPU nearest-neighbor folds, avoiding the original invalid qsort comparator. Native device kernels are unchanged. Dyadic inputs guarantee shared input bits; general point-set precision and the unsquared metric remain outside this named contract.

## Reference audit

Historical CUDA.jl smoke marker prints hardcoded PASS and no point-distance computation. Study reference implements genuine nearest-point loops and CAS maxima. Native CPU comparator is not a valid three-way ordering; study oracle uses direct max/min folds without altering GPU kernels.

## Numerical and performance evidence

Configuration `hausdorff_squared_a512_b768_dyadic_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T164718297065Z_hausdorff/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/hausdorff-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
