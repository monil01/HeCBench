# nosync: scientific and implementation observation

This is a backend-specific library-interoperability and hostexecution case, not a frameworkdevicekerneltranslation. The source has noauthoreddevicekernels: Thrust generatessequence/scans/reduces while an std::async hostthread overlaps API submission with explicitstreamscan. KA/JACC genuinely allocate devicebuffers, then call the same pinnedvendor CABI pipeline. Completeintegeroutputs and prefixfirstdifferenceidentity check realordering/dataresults, but do not prove actualGPUkerneloverlap or that par_nosync never synchronizes. Thepreallocatedtimedpipeline excludesoriginalmain outeralloc/free inall4models, retaininglibraryscratchallocation/hostthread/scalarD2H costs. Such results must beseparatefrom claims about portableauthoredkernelperformance.

## Reference audit

HistoricalCUDA.jl host-generatessequence andsequentially calls accumulate/sum ondefaulttaskstream, omitting nativeexplicitblockingcustomstream andstd::async hostfuture ordering. Outputs mathematically equivalent; mapping/hostexecutionpolicy difference, notS1. New namedinteropreference executes same nativeThrustpipeline with actualCUDA.jl allocation.

## Numerical and performance evidence

Configuration `nosync_thrust_stream_hostfuture1025_65535_preallocated_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T233602322034Z_nosync/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One programcandidate/validationprocess andoneNVCCvendorbridgebuild each;0authoreddevicekernelcompiler/JITchecks for allJulia models. No observedcompile/runtime/numericalrepair. Sharedsourcealgorithmextraction/referencework separate from frameworkallocationwrappers. Historicaloriginalattemptcounts unknown. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/nosync-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Library interoperability and measured boundary

All4 models use the same source-extracted Thrust calls and policies through `study_vendor_bridge.cu`, compiled with explicit legacydefaultstream. Native originalmain is also compiled unchanged underanalternateentry. CUDA.jl uses CUDA.zeros, KA uses KernelAbstractions.allocate, JACC uses JACC.zeros; actualframeworkdevicebuffers arepassedtoThrust. Thereare **zeroauthored/translateddevicekernels** and no claim ofpureportableKA/JACCscan/reduceimplementation. Thisvendorlibrarystratum is excluded from authoredframeworkkernelaggregateperformance. Thrustversionmacro200001wasreportedbyeverymeasuredprocess.

The named ROI is preallocated: it excludesoriginalmain device_vectorallocation/free, reset,streamcreation,fulloutputcopies, but includesinternalThrustscratch, hoststd::asynccreation/futureget andlastscanD2H. Contextwidecompletion bounds allmodels, so rawvendorlegacy-streamwork is included. Samealgorithm doesnotmake thesehostruntime/FFItimings evidencethatoneframeworkdevicekernel is faster. ActualGPUoverlap/par_nosyncnonblockingbehavior wasnotprofiled.

All20captures wereindependentlyaudited forcompletegeneratedsequence andinclusiveprefix values, reducedsum/lastscan/difference, plus exactfirstdifferenceidentity. Named1025/65535sizes are nonpowersoftwo andalltriangularsumsfitInt32; largest2147385345. Thisis exactnamedoutputcorrectness, not a generalintegeroverflowpolicy orallCLI/asynchronousstreamcertificate. Oneprogramcandidate/bridgebuild permodel, **0authoreddevicekernelJITchecks**.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
