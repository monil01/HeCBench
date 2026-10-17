# gru2: scientific and implementation observation

GRUoperatoraccuracyandthreadmappingaredistinct:oldscalarperhiddenJuliaisgenuinebutnotnativewarpreductionperformance. Explicitmappingalignmentfollowedbyfullsequence/hiddenandanalyticzero-weighttestsallowsfairerscope. KA/JACC directCUDAshuffleishybridbackenddependence,notpurecross-backendportabilityorframeworkunsupported. Namedodd/evenseqlayers/tailhidden/non-squareandpoisonedunusedpaddingtestbuffer/strideinvariants.

## Reference audit

HistoricalJuliaimplementsmathematicallygenuineGRUbutone-thread-perhiddensequentialdotsdiffernativewarp-perhiddenparallelshuffle. ExplicitP1referencealignmentrestoresnativewarp32mappingbeforetargettranslation;originalbytespreserved. NotS1wrongscientificoperator.

## Numerical and performance evidence

Configuration `gru_warp128_odd_even_multilayer_non_square_and_analytic_zero_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T212950757519Z_gru2/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One savedcandidate/onecompilerJITcheckforeachcurrentCUDA.jl/KA/JACCmodel,noobservedcheckfailure. Historicalgenuineone-thread-perhiddenimplementationwasP1mappingdifference,notS1. Explicitreferencewarp-mappingphasecostrecordedindependently; no historicalattemptcounts fabricated. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/gru2-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Independent recurrent dynamics and mapping alignment

All20artifactscheck24708exactFloat32inputslots(98832bytes)andall714sequenceoutputs/all-layerfinalhiddenvalues,independentFloat64reset-afterGRUrecurrencemaxerror 6.57917679225406e-08. Zeroallweights/biasescaseanalytic h_t=2^-t*h0 exactallmodels. General3layerseq3/B3/I37/H35exercisesoddhiddenpingpong,non-squareliveinputstrides,tailhiddenwarp,gateorderandpoisoned55.25unuseddeeperWihpadding;zero2layerseq4/B2/I5/H7exercisesevenhiddenparity. Sigmoidgatesin(0,1),convexhiddenblendandboundedhidden witness [{"min_gate": 0.3748079361422268, "max_gate": 0.6482282287230853, "convex_hidden_blend": true, "final_hidden_bound": 0.3285702908011578, "shape": [3, 3, 37, 35, 3], "zero_weight": false}, {"min_gate": 0.5, "max_gate": 0.5, "convex_hidden_blend": true, "final_hidden_bound": 0.0146484375, "shape": [4, 2, 5, 7, 2], "zero_weight": true, "analytic_h_t_equals_half_power_t_h0_exact": true}]. OriginalJuliawasmathematicallygenuinebutone-thread-perhidden,explicitP1native-warpalignmentrecordedbeforeportabletranslations. Nativekernel/orchestrationunchanged. CUDA.jlownshuffleframework_device_kernel;KA/JACCCUDA.shfl_down_synccalls hybrid_device_intrinsics,notpurecross-backendportability;noabsentfeatureinferredwithoutattempt. Hostviewcreationvsnativepointeroffsetsexplicitperformanceconfound. No backward/training/unmeasuredlarge-sequenceclaim.

AuxiliarysharedPythonwitnesshadoneB2parseerror(else64spacing)beforeCPUexecution;failedcandidate/log preservedandcorrectedCPUwitnesspassed. Thisisadditionalbenchmarkeffort,separatefromthreefirst-checkdevicekernelpasses. NoGPUrerunorchangedexecutionevidence.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
