# CRC64: interleaved GPU chunks and host GF(2) combine

## Operator and independent controls

Original Argonne/UChicago high-performance CRC64 source uses reflected ECMA182 polynomial0xc96c5795d7870f42, allonesinitial/finalxor (CRC64/XZ convention). Named65555/65677-byte deterministicinputs exercise64GPUworkers, differentchunkhead alignment/tails, and5×4-byte interleaved slicing tables. EverypartialCRC/size and all20actual timedAPIreturns pertrial checked exactly against independent Python bit-at-a-time GF2 oracle. Both append-check-byte residues are allones, both split/combine at12345 reproduce fullCRC, and untimed9-byte CPUfallback `123456789` gives995dc9bbdf1939fa. Full133895-UInt64captures include133358losslessactualinput/table/power/parameterprefixvalues,281observables and256exactguards. Saved audit checks all20artifacts and44boundsource/fixtureentries without converting UInt64CRC through Float64. Essential deterministicCPUfixtures/tables/oracle have generator/hash/provenance; tables/powers copiedexactlyfromsource, oracledoesnotusethem. Upstreamlicense retained in translatedhelper.

## Native scope and historical fidelity

HistoricalJulia touchesUInt8datawithxorzero andprintspass; no CRC, tablealgorithm orcombine. Preservedunchanged. Currentreference/KA/JACC implement the actual5-lane20byte algorithm, alignedUInt32loads via nonowninghostCuDeviceArrayview, byteheads/tails and hostGFcombine. Devicebody is ordinarytypedarray/GF2arithmetic, no foreignCUDAdeviceintrinsics. Entireimplementation is **mixed_host_device_pipeline**, authoreddevicecomponent **framework_device_kernel**; CUDA-specifichostbuffer/view setup declared. Little-endiantarget only, originalBigEndianbranchnotported/tested.

OriginalnativeAPI choosesfloor(nbytes/1024)workers up to24576 butlaunchesfloor(nthreads/64)blocks; source-static67584-bytewitness selects66butexecutes64workers, leaving2hostcombineentriesuninitialized. No actualGPUfailure/sanitizerclaim. Bothmeasuredsizesselectexactly64andarefullycovered, nativeunchanged. LargeMakefile32MBcase caps24576(divisible64), so not universallydefective. Julia studyAPI specializes64workers forthedeclaredsizes, notgeneraloriginalrandomCLIorNthreadselection; no originalfullprogramcertification.

## Timed stages and interpretation

Full originalparallelCRCAPI stages insideROI: fiveallocations, input/tablesH2D, GPUCRC, size/CRCD2H, dependentCPUcombine andfrees. Two per-callmeans summed, R10each. Julia CUDApool disabled withJULIA_CUDA_MEMORY_POOL=none, explicitsafeownerfrees, but hostcontainer/driverbinding/allocation/launch costs still differ. NotGPUkernel-onlylatency, notpureabstractionoverhead, notDRAMbandwidth. OriginalCLI CPUthreadclockMB/s notclaimed. Five checkedinprocessoutertrials afterfullAPIwarmup, wallclock/devicewidecompletion allmodels, average2–5 without dropping outliers. Actualsummedseconds: {"cuda": 0.00140278675, "cuda_jl": 0.000657289225, "ka": 0.0006300147, "jacc": 0.0006789522}.

Untimed guarded diagnostic invocation of the unchangeddevicealgorithm capturesactualdeviceinputs/tablebytes/64partialstates outsideROI; timedAPIbuffers are freed byoriginalAPI, so no impossiblepost-freeinputreadbackclaim. Sourcecopybindings and allactualtimedreturnchecksums provide provenance, notliteralreadbackofthefreedtimedallocations. Samefixture/code acrossallfour, allobservableoutputchecks are exact.

## Compiler loop and accounting

Currentreference one authoreddevicecandidate/JIT; firsthostprogram failed`CUDA.LLVMPtr`namespacebeforeauthoredJIT, correctedtoCore.LLVMPtr, secondprogrampasses. Twohostprogram/processchecks, onehostruntimefeedbackrepair, no authoreddevicecompiler/numericalrepair. Semanticrestorationofhistoricalalias separatefromfeedbackrepairs. KA andJACC eachonecandidate/kernel/JITevaluation,no repairs; shareauditedmath/hostdriver, JACCexplicitshmem0. One unchangednativebuildseries reused. Historicalcostunknown; prospectivephaseusagecontext/cache-inclusive/nonexclusive, broadworkflowoverlaps, savedsourceproxyincludeslicense/sharedcode. No generatedGPUoutputarchive inGit.

Finalbundle `.porting-state/portable_julia/runs/20261018T050611292219Z_crc64/result.json`, `study_config.json`, `saved_output_audit.json`, `native_thread_coverage_witness.json`, and `source_integrity_review.json`. PinnedJulia1.12.6/CUDAjl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,V100sm70. `make run` usesGPU6underthesharedlock. Coordinatorowns separatecommits andpaperintegration.
