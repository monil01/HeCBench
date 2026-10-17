# convolution3D: scientific and implementation observation

Despiteitsname, thenativebenchmarkimplements2Dvalidmulti-channelcrosscorrelation andcomparesthree3Dgridaxisorders, not3Dspatialconvolution. HistoricalCUDA.jlkernelsarealreadygenuineandretainedverbatim. Thefixtureexercisesbothspatialtailswithsignedintegerinputs, enablingexactindependentintegerdotwitnesses. Allthreegridmapsarepreserved; optionalcuDNNisnotenabled, andresultsdonotestablishfullmodelinferenceaccuracy.

## Reference audit

HistoricalCUDA.jl genuinelyimplementsall3nativegridpermutationsandarithmetic. Exactprefixkernels extractedunchangedfornamedharness; RNGinitialization alignedtoexplicitsharedfixture, nosemanticrepairclaimed. Nativebenchname3Dmeanschannel+2Dfilter, notthreespatialdimensions.

## Numerical and performance evidence

Configuration `conv_n2_c3_m5_h35_w37_k3_three_grid_orders_integer_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T185455234436Z_convolution3D/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/convolution3D-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.


Independent Python nested image/filter patch sums match all34650savedoutputs exactly in eachof20model/trialartifacts. This witnessusesmultidimensionalarraysratherthanthedeviceflatindexmacros. OriginalCUDA.jl kernelprefix is byte-for-bytepreserved andhashbound; nohistoricalsemanticfailureclaimed. Bothspatialtileedgesarechecked. This is2Dvalidmulti-channelcrosscorrelation, not3Dspatialfiltering; optionalcuDNN remainsunmeasured.
