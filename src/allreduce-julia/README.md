# Two-rank MPI all-reduce: named context-fenced study

Original CUDA files are unchanged and archived in `reference_history/`. An unchanged original RingAllreduce readinessprobe passes2realranks/GPUs at1/33/256/4099values. The measured nativeharness preserves sourcechunkorder/32×256gridstrideaddition and genuineMPIscatter-reduce/allgather, withpreallocateddevicebuffers andexplicitcontextcompletion. CUDA.jl/KA/JACC eachauthor anddispatchtheir ownadditionkernel. Shared `study_mpi_shim.cpp` wrapsMPIoperations only; no nativeGPUkernel iscalledfromJulia.

## Hardware and scope

Authorizedphysical5+6 only, undercentralGPU6lock; logicalrank0device0physical5 andrank1device1physical6. LauncherrequiresGPU5idle; MPIrankscheckforforeigncomputePIDs before/acrosstrials. DeferifGPU5hasexternalwork. UUIDs/libraryhashes/versions inenvironment_review.json. SeparatetwoGPUstratum, excludedfromsingleGPU/untouchednativeaggregates. Onlysamehost2ranks/CUDA-awareHPCX2.22.1(OpenMPI4.1.7rc1) measured; systemOpenMPI4.1.6 lacksCUDAawareness.

Sixsizes1,32,33,256,4099,1048576;3ringcalls/case. NonuniformrankdistinctFloat32dyadicinputs haveexactindependentsum1.5+(i%17-8)/16. Allactualinputreadbacks, bothrankoutputs/temporaryreceivedchunks andinput/output/tempguards checked.5,269,603Float32values/trial, exact2,106,001wordinputprefix; all20acceptedtrialcaptures matchbitwise. EmptysegmentatN1 exercised; fullN0/8M/64M/536Msourcecases/hugeiterationsunmeasured.

## Ordering protocol and retained failures

Source/currenttaskstream fences alone didnotestablishcompletefull-ringcorrectness: anearlier1Mcase failed despite exactinputs/guards. Failedbinary+SHA preserved in completion_protocol_review.json. MinimaldeviceSendrecv andIrecv/Send/Wait controls eachpass20exchanges/rank, so no genericMPIunsupportedclaim. The accepted separatelynamedcontext_fenced_v2 haswholecontextcompletion afterD2Dcopy, MPIreceive, GPUaddition andMPIallgather, symmetricall4models/freshfive-trialseries. This validates thatprotocol, notcausality or sourceMPIalgorithmdefect.

JuliaMPIalso required UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE toavoidUCXinterceptingJuliaGCsafepoints, andJULIA_CUDA_MEMORY_POOL=none forclassicMPIIPCcompatibility; UCX_MEMTYPE_CACHE=n alreadyset. See primary [MPI.jlknownissues](https://juliaparallel.org/MPI.jl/stable/knownissues/). Parser/terminalframing failures andJACCwrongdevice-streamglobalLaunchSpec areseparatelyarchived andcounted. No toleranceadjustments.

## Timing

Onefullringwarmup/case/rank outsideallfivecheckedtrials, mean2–5. ROI sums sixcase max2rankdurations of3ringcalls:MPIlengthAllgather, D2Dcopy, scatterMPI, ownGPUadd, allgatherMPI, fullcontextcompletion inside. Rankbarrier/max-timeMPIreduce, allocation/reset/capture/guardchecks outside. This preallocated stagedprotocol isnotoriginalallocation-inclusiveCLI throughput; no isolatedbandwidth orpureabstractioncausalcomparison.

## Fresh replay

Requiresnvcc12.0,g++,Julia1.12.6,CUDA.jl6.2.1/KA0.9.45/JACC1.4.0, CUDA-awareHPCX2.22.1. Sourceassets aretext; nobinaryfixture required. AbsoluteMPI/tool/projectpaths are recorded andmustberelocatedforanotherhost. SharedCABI/buildlibrary isgeneratedintemporarycwd. Do not useoldv1configstomeasurecurrentsources: theydocumentrejected/historicalstream-onlyattempts; restorecorrespondingarchivedsources ifreproducingthosefailures. Primaryconfigurationfiles:

```sh
python3 tools/verify_portable_julia.py --config src/allreduce-julia/study_config_native_context_fenced_v2.json
python3 tools/verify_portable_julia.py --config src/allreduce-julia/study_config_refs_context_fenced_v2.json
python3 tools/verify_portable_julia.py --config src/allreduce-julia/study_config_ka_context_fenced_v2.json
python3 tools/verify_portable_julia.py --config src/allreduce-julia/study_config_jacc_context_fenced_v2.json
```

Freshreplay removesreusefields orpointsnextconfigtofreshpriorresult. Historicalrawcaptures arelocal evidence, notcheckoutassets. Centralverifier ownslock; never nestflock aroundit. Standalonediagnosticlaunchers mustholdthatlock explicitly. RuntimeMPIenv/allocator/GCsettingdifferences aredeclaredinteropconfounds. Per-modelverification/accounting, paperobservations andlocaltraits/status/effort recordallcurrentprocess/candidate/tokenunits; cached/reasoningtokens aresubsets, notaddedcosts.
