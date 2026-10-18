# PSO-style source benchmark: scoped full-state verification

The source performs stochastic30Dposition/velocity andpersonal-best updates, but its `gBest` **accumulates all improved vectors withfitness<130**, ratherthan selecting the globalargmin. We preserve thisactual sourcebehavior, classify `native_behavior_known_scientific_defect=true`, andexclude canonicalscientific-fidelity aggregates. It is an optimization/atomic-update benchmark, notphysicalparticles or a validatedcanonicalPSOoptimizer. Fitness uses sourceytransform/phi3.1415f0 andis Levy-like, notclaimedcanonicalLevy. Inputsampler includes+1, actualrange[-5.12,6.12);72of900samplecoordinates exceed declared5.12, preservedratherthanclamped.

## Scope, numerical evidence and failure

Primary `study_config.json`/`20261018T003002862836Z_pso`:30particles×30dims, **teniterations/twentyoriginal-stylekernellaunches**, full900positions+900velocities+900pBests+30globalguide checkedsourceCPU1e-3, finite,actual2736inputprefixbyteexact. All20artifacts checked byCPU-only `audit_saved_outputs.py`; sourceCPUfullalgorithm is the trajectoryoracle, independentDoublefitness/marginalchecks supplement it ratherthanclaimindependentDoubleentireoptimizer validation. Personalbestfitnessnotworsethaninitial; guidefitness~6490.49 vsbestpersonal~24.557 witnessesnon-argmin behavior. NativeCPUmaxstateerror2.6702881e-5; all3Julia exactCPUonthiscase. EverytrialdirectCUDAandCUDAjlcomparisonpasses.

OriginalMakefile30particles/**10000iterations** was attempted first withthe SAME1e-3fullstate tolerance. Allfive native runs failed strongerfullstate checks:30positions exceedtolerance(max.00120544),3globalcoords exceed(max.0017395); velocitiesmax.000633, **all900pBestspass**(max.0001297), includingoriginalfirstparticle-onlycheck. `diagnostics/original10000_fullstate_failure.json` localizesstates; all5rawrejectedrecords/artifacts andsourceadapter retained. No tolerance relaxation, failedrecordpromotion or full10000trajectoryclaim. R10is a differentdeclaredinput, notthefirst10steps ofR10000:source rg seeddependsR-iter. Bothnative andallJulia use thesamecase. Failedseriesnotusedforpreferredtimings.

## Timed work and mapping

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|45.103µs/iter|29.535µs/iter|36.624µs/iter|33.887µs/iter|

FiveinprocessfullR10trialsafterfullwarmup,meanruns2–5. CompletehostCstdlibsrand/rand callsperiteration andtwokernel dispatches+blockingdevicecompletion aretimed, dividedby10. Julia usesccall tosamehostCstdlib RNG, notprecomputedoutsideROI. Statecopies/reset/CPUchecks/serializationoutsideROI. Four256threadmapblocks update900coords;one256threadbestblock updates30particles, inactive lanespreserved. This tiny/sparsehostdispatchregime and30atomiccontenders constrain generalization; no profileddevice-speedup/occupancycausalclaim.

Nativeprivatefloat[30]candidatearrays maptoJuliaMVector30Float32; own CUDAjl/KA+Atomix/JACC floatatomic APIs preserveguideadditions. Sourcearrays/privateMVector register/localspill loweringmaydiffer, actualcountersunknown. Noexplicitsharedmemory;JACCsets0B andseparate4/1blockLaunchSpecs. Allframeworkdevicekernels, noforeignCUDAdeviceintrinsics/operatorlibrary; hostCUDAallocation/completion andCstdlibRNG boundariesexplicit. Nativekernels/arithmetictargetunchanged.

## Effort and restart

HistoricalCUDAjlmarker/cannedPASS archived, restoredtwoactualkernels/sharedfitnesshelper. Reference/KA/JACC onecandidate/oneactualdevice-JITprocesscheckeach,two genericdefinitions,0targetfeedbackrepairs. Nativeoriginalfailedseries/R10validseries andtwoCPUgeneratorbuilds separate. Priorframework/reference/KAknowledge usedforJACC, no context-freefirstprogramsuccessclaim. Internalcompilerpasses/specializationsunknown. Sourcechars/4proxy separatefromprospectivemeter; usageincludescontext/cache/reasoning/coordination, broadworkflow/childwindowsoverlap, notadditiveexclusivebilledcost.

All3modelontology/scientificintent,failureevents,status,attempts,usage,fixtures/hashgenerators,STUDYandpaperobservation checkpointed. Rootcommits verifiedvariants separately. Nextqueueradixsort2.
