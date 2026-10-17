# crossEntropy: scientific and implementation observation

The benchmark uses a mixed-precision loss formula even for Float64 storage. The study preserves those narrowing points and checks every output in one complete original-size batch slice. Agreement establishes this kernel contract; it does not establish a canonical cross-entropy gradient or training behavior.

## Reference audit

Existing Julia reference has genuine loss kernels but converts all inputs to Float32 before target products and exponentials, unlike native Float16-rounded products and Float64 exponentials. Corrected storage-type product and narrowing sequence; this is a precision/mapping fidelity correction, not a proven wrong-output historical failure.

## Numerical and performance evidence

Configuration `crossentropy_bs1_w81_h8732_three_precision_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T171348723878Z_crossEntropy/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl and KA each passed one candidate/check. JACC retained one device candidate, but a host launch-constructor mistake failed before JIT; the corrected public constructor passed its second process check. This host runtime repair is separate from compiler feedback. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/crossEntropy-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
