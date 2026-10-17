# segment-reduce: scientific and implementation observation

The native benchmark uses general Thrust reduce_by_key over uniform runs. Portable versions use a genuine authored tree specialized to those uniform boundaries; this removes key-array reads and changes scratch allocation. All output keys and sums are checked across every native segment size, using signed nonuniform inputs so a constant-fill shortcut cannot pass. Performance comparisons include algorithm and resource choices.

## Reference audit

Historical CUDA.jl launches a one-element touch kernel; no segmented sums or output keys are computed. Study reference implements full parallel reductions for the native uniform-key contract.

## Numerical and performance evidence

Configuration `segmented_all11_n16384_signed_input_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T163949840908Z_segment-reduce/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/segment-reduce-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
