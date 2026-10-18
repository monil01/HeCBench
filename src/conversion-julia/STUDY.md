# conversion: scientific and implementation observation

Exact bit-pattern comparisons expose conversion semantics which decimal tolerances can obscure: signed zero, nearest-even half/BF16 ties, half subnormals, signed and unsigned narrow integers. All30 source type-pairs are retained. The original native host uses uninitialized input, so comparison uses explicit bit-backed shared fixtures instead. CUDA12.9 resolves the original half-to-byte header-overload ambiguity without source correction; that toolchain difference remains a performance covariate. These small cases test conversion fidelity and launch overhead rather than sustained memory bandwidth.

## Reference audit

Historical CUDA.jl executes a one-thread scratch touch for all30 advertised conversion labels; no source data is converted. Restore all30 actual typed casts with native mapping, initialized fixtures and exact bit evidence. CUDA12.0 half->char/uchar overload ambiguity resolves with unchanged source on12.9; this is native toolchain/header compatibility, not a target failure.

## Numerical and performance evidence

Configuration `conversion_all30_17_257_exactbits_10launch_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T042432893242Z_conversion/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One source candidate and whole-program GPU process per Julia model. Native unchanged source needed two buildchecks:12.0 rejected ambiguous half->char/uchar conversions,12.9 accepted; native toolchain compatibility event is not a KA/JACC or CUDAjl device candidate failure. Historical scratch-touch cost unknown. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective harness activity counters retained including cached/repeated context and marker lag, notexclusive billedcost. Historical usage unknown; sourceproxies partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/conversion-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Exact complete evidence

All20captures are bitidentical:8400exactinputprefix+8220complete rawoutputbitpatterns+120destinationguards=16740Float64-encoded words. All30directedtypepairs, sizes17/257,10timedlaunchmeans each, nativegeometrymin(n,256). Sourcebytepreservation andsourceguards checkedinsideeverytrial. Floatinginputs finite/inrangeforintegercasts; signedzero/halfway/subnormalcases covered, integernarrowingmeasuredmodulo. This is numerical representation primitive, no physicalmodel. Native source untouched; CUDA12.9 resolved oldheaderAPIambiguity, nativebuildvsJuliapinnedruntime12.0 confoundretained. One authored genericdeviceentry and30typepairs doesnotmean30compilerinvocations; observed1wholeprogram/JITprocesspermodel. Meansseconds {"cuda": 0.000195178825, "cuda_jl": 0.000254256, "ka": 0.0005826435250000002, "jacc": 0.00056340545}.
