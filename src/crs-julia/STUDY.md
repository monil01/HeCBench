# crs: scientific and implementation observation

Restoring parity semantics requires the actual Cauchy finite-field matrix rather than the historical synthetic XOR surrogate. Independent polynomial field arithmetic checks every packed native matrix column, parity output and all systematic k-row-subset invertibilities. Complete w-word groups make all shared-memory group neighbors initialized, including partial128-thread blocks. Portable inactive lanes remain at barriers rather than native early-return, without changing active data dependencies. Measured preallocated kernel pipelines include all20native variants/all50mkw and tasks1/3; source H2D/D2H taskcopy endtoend timing and GPU decoding remain unmeasured.

## Reference audit

Historical CUDA.jl uses synthetic byte XOR with j+m mix, no native GF Cauchy bitmatrix or shared64bit coding groups; MAX_K10 also differs native4. Restore actual source20kernel variants undergenericVal(m,w) with native128thread/block/group geometry. Fullyinitialized deterministic data and actualsourceCPUmatrices are explicit fixtures. Nativekernel and hostGFsource untouched.

## Numerical and performance evidence

Configuration `crs_all50_mkw_tasks1_3_groupcomplete_kernelonly_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T043326794987Z_crs/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

First saved candidate and whole-program GPU/JIT process for eachJulia model. Shared source-reviewed reference, native hostmatrix export and independent GF oracle reused sequentially forKA thenJACC. Observed activitywindow alsoincludes brief root-requested conversionontology precisionmetadata correction, explicitlynonexclusive. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective harness activity counters retained including cached/repeated context and markerlag, nonexclusive with priorconversionmetadata correction disclosed. Historicalcostunknown, sourceproxy partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/crs-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete finite-field evidence

All20artifacts bitidentical:94200exactinputprefix+61200allparitywords+400rawguards=155800UInt64words. Sourceinput/matrix preservation andmatrixguards selfchecked eachtrial. IndependentcarrylesspolynomialGF checks50Cauchymatrices,850systematickrowsubset invertibilities, everyparityword,50zerotaskparity cases/capture. Originalkernels/hostGFsource unchanged; sourceCPUexport native_matrix_columns.txt is essential fixtureprovenance. Pureframework ownkernelentry permodel,20requestedm/wvariants (notcompilerinvocations); CUDAjl/JACCdynamic1024B andKAstatic1024Bshared, JACCpublicshared128wordtemplatepreload retained. NoCUDAdeviceextension. Kernel-onlyROI explicitlyexcludesoriginalH2D/D2H taskcopy performanceandhostGFsetup. Meansseconds {"cuda": 0.0013589242500000004, "cuda_jl": 0.0023442697500000004, "ka": 0.0029624950000000008, "jacc": 0.004270033249999999}.
