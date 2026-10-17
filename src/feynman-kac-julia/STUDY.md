# feynman-kac: scientific and implementation observation

This port retains the native stochastic elliptic-PDE estimator, including its common per-workitem seed. Complete native outputs are only an interior count and analytic-solution squared-error aggregate. A reduced named grid and path count enable full semantic checks, but agreement is not evidence of sampling independence or convergence. The analytic reference is part of the source contract.

## Reference audit

Existing CUDA.jl implements genuine per-point random walks, Park-Miller recurrence, potential integration, and atomic error reduction. Measurement helper extracts the same algorithm; no established historical semantic defect.

## Numerical and performance evidence

Configuration `feynman_kac_ni17_nj9_paths32_h001_seed123456789_r3_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T171720642780Z_feynman-kac/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One unchanged device candidate per model. CUDA.jl and KA each encountered a shared helper parser error before device compilation, then passed their second process check; one shared measurement-helper repair, not two independent algorithm repairs. JACC sees the corrected helper. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/feynman-kac-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

Evidence strength: `aggregate_count_and_error_sum`. No pointwise path or estimated field comparison is performed. Native analytic RMS on this fixture is0.0345829384; no convergence or confidence claim follows.
