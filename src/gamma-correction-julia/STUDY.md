# gamma-correction: scientific and implementation observation

Named colored64x64BGRA fixture exercises weighted luminance and alpha semantics missing from historical single-channel Julia code. All output bytes are compared exactly. Host transfers reset in-place image before every kernel and are excluded symmetrically; each invocation synchronizes as native original loop.

## Reference audit

Historical CUDA.jl operates one grayscale byte and drops RGB luminance weights, BGRA struct layout and alpha output; study reference now matches native four-channel operation.

## Numerical and performance evidence

Configuration `bgra_colored4096_gamma2_reset_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T161114923990Z_gamma-correction/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One device-kernel candidate for each model; CUDA.jl and KA each required two process checks because a shared host fixture comprehension had a B2 parser error, repaired before numerical trials. JACC inherits the repaired fixture and requires one check. Shared measurement repair is separate from device-kernel compiler repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/gamma-correction-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
