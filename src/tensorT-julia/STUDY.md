# tensorT: scientific and implementation observation

The study preserves the native shared-memory transpose, including its5900word capacity and Float32 reciprocal indexing, on smaller six-dimensional extents. Every output is checked through an independent six-index permutation. Float64 is payload storage, not floating-point arithmetic precision; performance cannot be interpreted as Float64 throughput. JACC shared initialization copies a full template in addition to native tile loading.

## Reference audit

Historical Julia reference only launches a one-word smoke marker and prints hardcoded PASS, with no tensor permutation. Genuine reference now preserves native tiled six-dimensional permutation and reciprocal indexing.

## Numerical and performance evidence

Configuration `tensort_dims5_7_11_3_4_5_shared5900_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T171954276374Z_tensorT/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/tensorT-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
