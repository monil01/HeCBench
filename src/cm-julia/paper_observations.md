# CM research observations

Configuration `cm_U133A22283_sig31_R4099_two_drugsets_full3GPUstages_hostfeedback_v1`; full gene universe with a named resident statistical pipeline.

## RQ1: fidelity and scientific guarantees

All20 complete361,102-value Float64-encoded captures are bitidentical, with352,282 actual input/prefix values. All8,198 random scores and416 reduction partials plus14 host statistics are independently checked per capture. The original CLI also succeeds on the generated22,283-gene/three-profile dataset with100,000 random signatures, with independent checks at its printed precision.

The original helper returns0 for50,000/100,000 exceedances, whereas the two-tail control gives1. That defect is preserved, including actual target-host diagnostics. The primary4,099-sample contract cannot encounter exactly50%; it is not a certificate for the full statistical domain or biological application.

## RQ2: feedback and observed effort

CUDA.jl and JACC pass first candidates/checks. KA takes two program checks: dot and score GPU warmup succeed first, then a count! host constructor MethodError; renaming the entry fixes dispatch without changing arithmetic. Count two device-JIT processes, one arithmetic body candidate, one host API repair. Shared native/RNG readiness and verification/annotation remain separate. Actual token windows are context/cache/session inclusive and ordered helper reuse applies.

## RQ3: execution times

| Version | Mean2–5 (µs) |
|---|---:|
| cuda | 79.239 |
| cuda_jl | 206.346 |
| ka | 269.792 |
| jacc | 323.402 |

ROI includes six source GPU launches, partial dot/count D2H and dependent host normalization/statistics, with preallocated resident inputs. IO, RNG, setup/H2D and final score capture are excluded. It cannot be compared directly to original whole-CLI walltime.

## RQ4: mechanism evidence; RQ5: selection

All target GPU components are framework-only. The full timed implementation is hybrid host control because computed dot/count readbacks determine dependent work/results. KA uses static512B tiles, the others dynamic512B; JACC adds128-value zero-template preload/barrier before complete partial overwrite. These and host wrapper/transfer API costs are covariates, not isolated frontend causes. No measured roofline or occupancy claim.

See source-bound JSON records for scope, numerical qualifications and usage windows.
