# channelShuffle: scientific and implementation observation

Channel shuffle has exact data-reordering semantics but different memory mappings. Historical CUDA.jl NHWC removes native shared staging; KA preserves a static shared tile, while JACC uses dynamic shared storage with an extra template initialization. These are observable source-level confounds, not evidence that one abstraction intrinsically requires more traffic. Both layouts retain native launch geometry for the named entry.

## Reference audit

Historical CUDA.jl outputs are numerically faithful. Its NHWC kernel directly reads global inputs instead of native shared staging; retain this actual mapping as an explicit performance confound. New reference changes only fixture/measurement harness.

## Numerical and performance evidence

Configuration `shuffle_n4_c32_g4_hw64_bothlayouts_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T162849202248Z_channelShuffle/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

KA candidate1 invoked @localmem with a single call expression instead of separate type/dimension arguments, producing a B2 host macro MethodError. Candidate2 repairs the invocation; native and CUDA.jl valid trials reused. JACC begins only after KA passes. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/channelShuffle-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
