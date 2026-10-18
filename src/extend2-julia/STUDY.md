# extend2: scientific and implementation observation

The historical Julia translation already preserves a meaningful domain operator. Numerical checks therefore separate source-specific alignment semantics from textbook labels and from inherited timing/memory-layout differences. Independent row-snapshot integer recurrence agrees with all17shipped6outputgoldens and five deliberate match/mismatch/gap/band/zdrop fixtures, then fullworkspace checks expose errors hidden by final scores. This is a serial dependency-DAG benchmark, not evidence about scalable parallel alignment or biological mapping quality.

## Reference audit

Historical CUDA.jl genuinely implements native serial banded affine-gap extension, retained arithmetic/controlflow. Its separate H/E globalarrays differ from native eh_t AoS; mapping preserved and disclosed. Source native offload timer includes allocation/copies/free whereas historical Julia timer covers kernel; new namedpreallocatedkernel-only scope matches allfour and does not claim originaloffload speedup. No algorithm repair/nativecorrection or observed reference numericalfailure.

## Numerical and performance evidence

Configuration `extend2_17golden_5synthetic_fullscratch_serial_kernel_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T054549783842Z_extend2/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Retained historical genuineJulia arithmetic/controlflow; onecandidate/process permodel, no current observed compiler/runtime/numerical repairs. Reference phase covers typedguardedharness/nativeadapter and independentDP/fixtures; historical translationcostunknown. OrderedKAthenJACC reusepriorverifiedcontext. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Prospectiveharnessactivitycounters include cached/repeatedcontext andcoordination; nonexclusive, historicalusageunknown andsourceproxiespartial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/extend2-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete exact sequence-extension evidence

All20artifactsbitidentical25334Int32words:6799exactinputprefix+fullquery/target/matreadonlyvalues+10746result/scratchoraclevalues+1232guards. All132resultstatistics (6percase) and completeH/E/queryprofile agree independentrow-snapshotDP exactinteger and17externalgoldenrecords+5syntheticcontrols. SourceM-onlygap-opening, asymmetricalpenalties, rightmostrowties, strictglobalimprovement, band/zdrop preserved, notgenericSmithWaterman substitute. ExistingJulia genuinebodyretained withSoAH/E insteadnativeAoSeh_t, memorylayoutconfound recorded. Namedpreallocated22kernel-onlyROI excludes originalnativealloc/copy/free offload; nooriginalendtoend speedupclaim. Onecandidate/JITprocess eachmodel, zeroobservedrepairs; JACCexplicitshared0, allframeworkownserialkernel1thread. Meansseconds {"cuda": 0.03451636, "cuda_jl": 0.043363394500000006, "ka": 0.045405436500000014, "jacc": 0.04770107325}.
