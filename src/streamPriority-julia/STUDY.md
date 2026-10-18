# streamPriority: scientific and implementation observation

This workload separates device translation from CUDA-specific host orchestration. All four actual computations are preserved, and every final buffer is checked against both references and an independent Float32 recurrence witness. Real prioritystreams and reused dependency events preserve source scheduling. The source priority timer finishes training without waiting for finalprefetch, whereas default timer waits allwork; pending background work is queried aftertimer and completed beforechecks. Thus per-modeframework comparisons have matched boundaries, but default-versus-priority throughput or causalprioritybenefit claims are unsupported. The source is a synthetic training-like demonstration, not a validated learning model.

## Reference audit

Historical CUDA.jl prints hardcoded priority range, PASS and timings without executing any training/priority pipeline; its timed_dummy helper is unused. Preserve original bytes; new reference implements all four source computations and real default/priority streams, events and dependencies. Native main defines/calls only ML-training simulation; all source device/reference functions audited and unchanged.

## Numerical and performance evidence

Configuration `streamPriority_all_native_usecase_two_modes_fullstate_source_completion_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T013405125984Z_streamPriority/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Historicalsmokestub S1 corrected separately. One authoreddevicecandidate permodel, fourdefinitions. CUDAjl firstprogramcompiled/executed all4 and emittedfivePASS, thenSIGSEGV at exit because agent manually unsafe-destroyed events without unregistering managedfinalizers. All5captures rejected fornonzeroexit and retained. Sharedinfrastructure R1 hostlifecycle repair usesBase.finalize; secondprogram exitscleanly. Authoreddevicekernelcandidate unchanged; no targetkernelcompiler/runtimefeedbackrepair. KA/JACC adoptvalidatedhostharness and each haveoneprogramcheck. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/streamPriority-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Source usecase, timing boundary and actual computations

Actual main defines/calls only the synthetic ML-training usecase. All four native kernels,256iterations and256thread geometry remain intact. Prefetch is device arithmetic/buffer generation, not hostI/O. Allthree Julia pipelines are `hybrid_host_control`: ownframeworkdevicekernels plus explicitCUDA.jlprioritystreams/eventwaits; KA/JACC call no CUDAdevice intrinsics. No actual lossgradient/networktrainingaccuracy or crossvendorhostportability claim.

Real device range and actual streampriorities are retained in `stream_completion_evidence.json` along with every after-timer pending-prefetch query. Default timing context-completes allwork; priority timing waits only training. Prefetch is then completed outside the timer beforefullbufferchecks, symmetricallyall4. Backgroundwork was observedpending atpriorityboundaries. These are matchedper-modeversionmeasurements but do not support default-versus-priorityequalworkthroughput or causalpriorityspeedup. Freshfirstfetch wait follows sourceunrecordedevent semantics; subsequent recordedforward/fetch dependencies are real.

Every acceptedcapture checks92,180finalvalues acrossprefetch,weights,gradin,activation,gradout and92,220exactinput/controlvalues. The independent integerstage/pair-Float32recurrence witness verifies all20captures, zero-gradientunchangedweights and nonzeroSGDupdates. Source1e-3bound retained; fullpointwiseoutputchecks exceedoriginalgradout-onlyvalidator. `per_component_performance.json` preserves4component×5series/meanlast4.

The first Julia program emittedfivePASS captures butlaterSIGSEGV fromagentmanualunsafe-event-destroy plusregisteredGCfinalizer. Verifier rejectedallfive fornonzeroexit; rawlogs/captures retained asdiagnosticsonly. InstalledCuEventconstructor/destructor hashes/callflow are in study_notes. ManagedBase.finalize fixes hostlifecycle; unchangedauthoreddevicecandidate was revalidated insecondprogram. This sharedinfrastructureR1 belongs to referencehostorchestration, not fourfaileddevicecandidates, and no first-program-successclaim is made. Root limitedheader/stub/Makefile exposure beforetransfer is disclosed; usagewindows cover activityincludingverification/metadata, notexclusivecodingcost.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

## Scientific family review

Primary family follows the actual synthetic stream-priority/overlap scheduling workload. Training-like labels and an SGD-like pointwise update do not establish a defined model, loss, or scientific learning operator; machine learning is secondary advertised application context only. No convergence or training-quality claim. This annotation changes no measured code, timing region, numerical evidence or archived configuration.
