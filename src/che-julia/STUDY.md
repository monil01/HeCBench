# che: scientific and implementation observation

The native binary-mixture coefficients reduce mathematically to a quartic double-well potential. An independent implementation uses that reduced expression and periodic neighbor-difference sums, while GPU translations retain the original coefficient arithmetic. Complete four-field comparisons plus a separate periodic concentration invariant test more than native array agreement. Constant and Fourier-mode inputs supplement signed dyadic fields and partial tiles; neither short-step accuracy nor conservation establishes long-time coarsening or continuous PDE convergence.

## Reference audit

Historical CUDA.jl only executes a one-value smoke marker and prints zero-iteration timing; no native Cahn–Hilliard computation. Preserve original bytes. Restore periodic chemical potential, local free energy, conserved concentration Euler update and elementwise swap with matching source thread geometry. Native device bodies unchanged; named compiletime dimensions and preallocated kernel-only region explicitly exclude original diagnostic D2H/integrals/file I/O.

## Numerical and performance evidence

Configuration `che_periodic_16x8x8_19x9x7_steps1_3_varied_constant_fourier_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T040123092045Z_che/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

First saved candidate and whole-program/JIT process passes for each Julia model; no observed target repair. The historical smoke reference is S1, not a historical compiler failure. Reference/oracle work is shared before KA then JACC. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Actual prospective harness activity windows retained by phase, including cached/repeated context and marker lag; nonexclusive. Historical usage unknown and source proxy partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/che-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete independent evidence

Every capture:4269exactinputprefix+17076completefieldvalues+32rawguards=21377Float64values. Independent rawbyteguard/prefix, fsumperiodicmass and analyticconstantstate checks pass all20captures. Maximum independent mass drift 2.7755575615628914e-16. Source genericJuliaentrydefinition1 is specialized by4Valtags; native has4authoredkernels, allmodels execute40calls per4-case trial. No shared memory or foreigndeviceintrinsics. Means seconds {"cuda": 0.000161292, "cuda_jl": 0.00028701050000000004, "ka": 0.00043723675, "jacc": 0.00055821725}. Native includes original kernelbodytext underdimensionnamespaces; no256cubed/longcoarsening/diagnosticI/O timingclaim.
