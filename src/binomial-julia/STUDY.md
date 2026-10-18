# binomial: scientific and implementation observation

A financial tree operator permits an independent terminal-probability witness instead of another backward recursion. All145 individual prices are checked against exact serialized coefficients; up to0.00396 difference from continuous Black–Scholes remains a separate discretization/coefficient observation, not a language mismatch. Native AoS, Julia SoA and JACC shared-template initialization are explicit performance covariates.

## Reference audit

Historical CUDA.jl faithfully implements source 2048-step segmented tree with 128 threads, 16 private values per thread and 129-value shared exchange. Study preserves device body, adds exact serialized preprocessed inputs, full per-option expectation oracle, guards and five checked trials. Native AoS and historical Julia SoA layouts remain distinct; native __expf versus Julia exp is retained under predeclared Float32 tolerance. Original 1024-option/offload-inclusive CLI is unmeasured.

## Numerical and performance evidence

Configuration `binomial_2048step_original_exchange128_options16_129_repeats1000_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T034005211493Z_binomial/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Reference one authored device body, two whole-program checks after a shared parser repair. KA two frontend candidates, first explicit return rejected by @kernel before device JIT; no device-compiler rejection. Successful reference/math/fixture work is reused by KA then JACC. KA candidate2 reached GPU warmup but its escaping MVector exhausted dynamic GPU memory; candidate3 uses documented @private per-item storage across barriers and passes. KA one frontend feedback repair is included in compiler_feedback_repairs; device_compiler_feedback_repairs remains0. JACC first candidate/check passes with its public shared-template API. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective actual perphase harness activity counters retained, including repeated/cached context, waiting/diagnosis and marker lag; nonexclusive costs. Historical usage remains unknown; source proxies partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/binomial-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete scoped evidence

Twenty captures each contain1450exactFloat32inputprefix values,145callprices and4exactoutputguards. Independent raw-byte audit confirms all20prefixes/guardgroups exactly. Allinputallocations also check guards exactly in each harness; native AoS and Julia SoA guard layouts differ and those inputguards are not serialized. Independent direct terminal-binomial expectation does not share backward recurrence; max continuous Black–Scholes difference0.00396 is reported separately, not a language error. KA candidates: B2 explicitreturn frontend rejection; R1warmup dynamic allocation of68-byte MVector; documented @private storage passes. JACC firstcandidate passes. No device compiler rejection or unsupported feature. Mean per2-case kernel evaluations (1000repeats each): {"cuda": 0.0005282354855000001, "cuda_jl": 0.00069736783675, "ka": 0.0006883462605000001, "jacc": 0.000693922677} seconds.
