# CC: degree-adaptive connected components

RQ1: All20 saved37916-Int32 captures are bitidentical, including33559 actualdevice-inputprefix values and every final componentminimumID/guard. Independent BFS reconstructed fromactualCSRreadbacks passes. All60 untimed stage captures satisfy monotone samecomponentparents and exactfinalBFSlabels; worklist ordering is not claimed identical. Original synthetic fullCLI passes; no native repair.

RQ2: CUDA.jl2 device-reaching candidates/checks, one1-based indexed-shuffle API correction; KA/JACC1 candidate/check each. Five sourcealgorithm definitions plus oneuntimed parameterobserver pertarget; shared device/host helpers authoredinCUDAjl and reused. Prospective per-target counters retain inheritedcontext/cache and ordering. Sharednativeprep, verification andannotation separate. Supplemental diagnostic stringtransport failedbeforeJIT once and is separate infrastructure effort.

RQ3: Mean2–5 composite seconds CUDA0.0002266643475, CUDA.jl0.002185381295, KA0.0020514413275, JACC0.0022095395. Sum3 per-graph R100 full5stage pipeline means, warmed exacthostbatch; transfers/setup/checks excluded. The9–10x gap is observational, without a measured cause.

RQ4: Native fourdeviceglobals become a4entrydevicecounterarray. Native launch_bounds256/min8blocks hint is not explicitly replicated inJulia. InstalledCUDAjl atomicRMW/CAS lowering uses acquire_release/acquire; native generated ordering, registers and occupancy unprofiled. Allversions640blocks×256threads sourcehardwaregrid. CUDA.jl is framework_device_kernel; KA/JACC explicitlyhybrid usingCUDA CAS/shuffle/staticshared/barrier. No roofline coordinate is inferred fromthis sparse/integer pipeline.

RQ5: Full degree-adaptive sourcealgorithm is expressible throughKA/JACC plusCUDA extensions. These results do not establish framework-only/cross-backend portability or general graphscale performance. Native is fastest for these synthetic cases; targetchoice must retain backendextension andresourcepolicy boundaries.
