# mallocFree: actual runtime lifecycles, no numerical outputs

The source is a host CUDA-runtime allocation/free microbenchmark. It has zero device kernels and no numerical outputs. Historical Julia only printed fake zero times without CUDA API calls; reference_original preserves that stub. New entries restore actual runtime calls, not fabricated array checks or an empty kernel.

## Full source allocation workload and evidence

All nineteen source sizes64B..16MiB, three types(cudaMallocManaged with default attachGlobal1, cudaMalloc, cudaHostAllocMapped2), bulk500 allocations followed by500 matching frees are retained. Maximum simultaneously allocated bytes8,388,608,000(about7.8125GiB; sourcecommentrounds8G), withinV10032GB and availablehostRAM. No framework memory pool substitutes these APIs.

Every model has five checked in-process lifecycle trials, mean2–5. Each trial retains57 complete component status records with500 allocation return codes,500 non-null flags and500 free return codes. lifecycle_audit.json independently checks all20trials:570,000 successful allocations and570,000 matching frees, plus every non-null flag. checked_values0/errorsnull/noartifact correctly mean no numerical-result claim. These cases are execution_only_verified and excluded from numerical-port counts. Successful allocation/free does not validate memory contents, mapped-device loads, migration or coherence.

## Actual API implementation and timing scope

A narrow source-backed C++batch routine calls dynamically resolved entries from exact libcudart.so.12.0.146(version12000). The native adapter embeds that routine; Julia wrappers call its shared CABI DSO. Allfour actual loaded-runtime paths/hashes agree in runtime_provenance.json, resolved from function addresses rather than link arguments. The host helper is compiled with actualg++13.3.0, recorded source/binary/build hashes. The initialreferenceconfig's g++12.3 description was a typo, corrected hash-bound in study_adjudications.json; measured code/source/times never changed.

Each component time is the host allocation or free loop divided by500, matching the sourcebulk mean. Total sums114componentmeans; actual whole batch wall duration is roughly500times that sum. Status/non-null/pointer reset stores are added identically inside the instrumented loops; serialization/check branches are outside. Common C++timers exclude Julia/package/entry-call overhead, so timings cannot measure abstraction frontend costs. Nativeembedding versus PIChelper, contextstartup, OS/pinning/allocatorstate and unresolvedactualdriverDSO are explicit covariates. ActualruntimeentryDSO identity does not prove identical wholeloaderstate.

Mean total milliseconds(sum114per-callmeans):CUDA47.620443, CUDA.jl34.065406, KA41.324742, JACC34.765960. Component_timings.json retains all114meancomponents and everyoutertrial; no fastestselection or guessedcausalexplanation. One warmallocation/free pertype/size precedesfivebulkseries; originalcoldtestInitlatencies are unmeasured. The sourcecoldmapped-freeprint uses staleallocationduration, outside this bulk timing contract. Unused CPUsetupA initialization is outside source timers and excluded.

## Ontology and effort

Allthree versions are vendor_library_interop, with zero authoreddevicekernels and zero actualdevice-JITchecks. KA/JACCpackage entries provide explicit CUDA-runtime orchestration and do not invoke @kernel/parallel_for or frameworkallocationAPIs. This is not an independently implemented KA/JACCkernel or evidence a feature is unsupported. Runtimeallocation has no floating-point roofline position.

Each model passed one hostprogram/processcheck, no compiler/runtimefeedbackrepair. The sharedreferencephase restores realhostAPIcalls from the historical zero-printstub, builds one CABIhostlibrary and one nativehostharness; no GPUkerneltranslation is counted. Generic verifier event labels mentioning deviceJIT do not establish device compilation here; explicit effort records set zero. Historical attempts/consumption remain unknown. Thinentrysource proxies, sharedCABI/drivercosts and context/cache/coordination-inclusive overlapping usage windows remain separate, nonexclusive measures.

## Reproduction and resumption

Run python3 src/mallocFree-julia/build_runtime.py once to create the local ignored hostDSO, then verify study_config.json. Completed final result20261018T081442421808Z_mallocFree; refs/KA configs preserve exactunchangedseriesreuse. No numericalbinaryfixtures orGPUoutputarchives exist. Source_execution_proof.json, NOTICE.source and reference_original retain source/license/fidelity boundaries. Root alone commits each version separately; canonicalguides/workercheckpoint resume with minisweep.
