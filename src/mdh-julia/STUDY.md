# mdh: scientific and implementation observation

The finite study evaluates a screened molecular potential, preserving three native shared-layout/fast-exponential variants and four gridpoints per workitem. Independent Float64 CPU folds check every potential; positive charges additionally require positive finite results. Native default sets every atomic size to5877, making its exponential argument exceed Float32 range even for maximal unit-cube separation; the original comparison can accept inf-inf NaN differences. This named finite physicalparameterfixture preserves the kernel equation and avoids that invalid default evidence, without asserting full APBS physical validity.

## Reference audit

Historical Julia computes the screened-potential formula but aliases all3 native shared-layout/fast-exp variants to one unshared loop. Restored genuine AoS/sharedSoA and fast-exp variants; P1 mapping fidelity, not established mathematical wrong-output. Historical default permits matching infinities.

## Numerical and performance evidence

Configuration `mdh_finite_natom129_grid1024_wg64_threevariants_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T173703711246Z_mdh/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One devicecandidate per model. Shared host-oracle generator parser error failed before JIT for CUDA.jl and KA; replaced with explicit nested loops, one measurement-helper repair. JACC sees the corrected helper. Device candidate1 then failed inference on a closure capturing branch-assigned atom temporaries. Candidate2 expands the same four lanes explicitly; no vector-width or algorithm substitution. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/mdh-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
