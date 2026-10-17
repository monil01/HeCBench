# openmp: scientific and implementation observation

This benchmark studies concurrenthostsubmission, so the primaryregionincludeshostpartitioninitialization,allocation,H2D,devicework,D2H,free,andjoin. Allsixpartitioncountsandbothoutersweepsarechecked. Juliahosttasks adaptOpenMPOSthreads; extraJuliahosttemporaries andruntimescheduling arevisibleconfounds. Frameworkdevicekernelcostcannotbeisolatedfromtheseendtoendnumbers. NoactualGPUoverlapclaimismadewithouttrace.

## Reference audit

Historical Julia only launches a one-word marker and prints12hardcodedPASS messages; no arrayincrements or hostconcurrency. Genuine devicearithmetic/reference repaired. JuliaTasksadaptOpenMPthreads as a separate documentedP1 hostmappingchoice.

## Numerical and performance evidence

Configuration `openmp_hostconcurrency_n262144_f1to32_repeat37_legacyalloc_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T174640140333Z_openmp/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl/KApassed firstcandidate/check. Nativebuildfirstfailedmissingreference.h; reconstructed ownedhostchecker withoutchangingnative devicekernel, separate nativehostdependencyrepair. JACCinheritsverifiedsharedhelper. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/openmp-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
