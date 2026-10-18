# hypterm: scientific and implementation observation

A radius-four stencil footprint does not establish eighth-order scientific accuracy: actual source coefficients have third moment0.0039998963. The study preserves those coefficients and distinguishes exact discrete-operator agreement from continuum accuracy. Constant manufactured fields check zero divergence; affine density/pressure at constant velocity provides an analytic first-moment check, and spatially varying positive EOS-consistent fields exercise all five nonlinear conservative fluxes. Historical Julia was a genuine port; matching source Float32-promoted coefficient policy is precision alignment, not invention of missing physics.

## Reference audit

Historical Julia genuine3-stage hyperbolic flux kernels retained as original evidence. Study aligns native Float32 literal coefficients and dx rounded once then promoted to Float64; oldJulia literals were full Float64. Mechanical source assignment expressions preserve pressure grouping and source energy x-z-y accumulation order; no S1 smoke/algorithm substitution or native correction. Source cubic16x4x4ceilgrid and five flux fields audited.

## Numerical and performance evidence

Configuration `hypterm_cubic12_17_24_3stage_F64_sourceF32coeffs_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T065613354311Z_hypterm/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One authored device source candidate each. CUDAjl whole-program1 stopped in shared host parser beforedeviceJIT (missing generator space B2); process2 first actualdevicecompilation passes. Sharedhostcandidate archived/originshared, not frameworkunsupported/compilerbodyfailure. Source-derived common bodies and independent oracle prepared in reference phase, then KA, then JACC sees validatedKA. Prospective windows include cached/repeatedcontext, coordination/waiting and same-benchmark ontology/metadata preparation, not exclusive billedcodecost; workflowoverlapssubphases do notsum. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective activity windows include cached/repeated context, same-benchmark ontology/metadata/waiting and coordination; historical consumption unknown, source-only proxies partial and omit shared code.

Reproduce: `python3 tools/verify_portable_julia.py --config src/hypterm-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific accuracy versus stencil footprint

Actual source Float32-promoted coefficient first moment1.0000000037252903, thirdmoment0.0039998963475. Radius4 alone is not evidence of eighth-order convergence; no violation of an explicitly promised native order is asserted. Affine independent actual-stencil analytic error6.776263578e-21 and separate continuum coefficient-rounding error2.239630653e-13 are retained. Constant state has exactlyzero divergence. Positive manufactured idealgas states usegamma1.5; this validates a local discrete flux operator, not a complete evolved fluid solution.

All20 captures430095Float64words:163738exactprefix + actual8readonlyfields/all5fluxfields/all312canaries. All102325 outputvalues, fullzero halos and constantcase, independentconservativefluxoracle pass in everycapture; Julia3 bitidentical, native/Julia maxabs6.776263578e-21. GPU allFloat64, Float32onlythe compile-time source literal rounding policy. Source16x4x4full/partial3Dtiles retained; JACC pinned public3D API supports requiredgeometry withshmem0. NativeoriginalD308CLI/unmeasurednoncubicparameters remain outside certifiedscope. Aggregate mean2-5 seconds {"cuda": 9.73103e-05, "cuda_jl": 0.00015589252500000002, "ka": 0.000240791525, "jacc": 0.000283166975} sums9separatelycompletedstageROIs, includinghostlaunch/completion; no device-onlythroughput or causalrooflinebottleneck claim. Three entrypoint types are separate from actual uninstrumented compiler invocation counts (unknown).
