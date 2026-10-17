# memtest: scientific and implementation observation

Nativeperformanceismeasuredonlyfortest5;retainitsfullthree-kernelchainratherthanclaimingallmemorydiagnostics. Completebufferoracleanddeliberatepairfaultwitnesscheckdifferentproperties:patternmoveanddiagnosticerrorrecord. Named2MiBcasewithoriginalgridisnotamemorybandwidthsaturationorhardwarehealthstudy. Untimednativeaddress-pattern/walking-address/movinginversiontests0–4remainunmeasured;historicalJuliaoffsetaddresssubstitutionisnotcertified.

## Reference audit

Historical test5 init/move algorithm genuine, kept verbatim. Historical diagnostic stores element offsets instead of actual device addresses and copies first read as second read; corrected actual-pointer and second-read semantics for the deliberate fault witness. Its CUDA.@atomic already returns the old value under pinned CUDACore, so no slot-index defect is claimed. Untimed test1 offset rather than address mode is NOT certified.

## Numerical and performance evidence

Configuration `memtest_native_timed_test5_2mib_allbuffer_faultwitness_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T194406045200Z_memtest/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Reference measurement/diagnostic repair and both target translations have independent process/candidate records. Original test5 init/move bodies are retained verbatim; actual diagnostic pointer/reread semantics repaired. Test0–4 not measured. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/memtest-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Diagnostic contract and scope

This configuration measures only native timed test5, on 2 MiB rather than the original 2 GiB. All 524,288 UInt32 words and the zero error counter match the independent closed-form word-source oracle in all 20 artifacts. An untimed injected first-pair fault checks count=1, actual device address, expected value, current value, and actual second read in every model. Tests0–4 remain unmeasured. The retained 65,536-block launch has only two active move work items; neither bandwidth saturation nor hardware health is established. Bounds checks and compiler decisions are unprofiled; the timing gap is an observation without causal attribution.

Reference-preparation usage boundary: a small final metadata qualification was written in the END tool call and may fall outside that child interval. Raw marker lag is retained; the parent workflow still includes that work. No retrospective counts were invented.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
