# bh: scientific and implementation observation

Restored native ECL-BH4.5 pipeline: initializationonce then8stagesperstep (boundingbox,clear1,atomicCASoctreeinsertion,clear2,volatileCOMsummarization,octantDFSsort,warpcollectiveBHforce,integration). Full10publicFloat32statecomponents; no denseforce substitution. Original24baseblocks and256threads, stageblockfactors2,1,2,1,1,1,2,2 (clearstages24blocks); nodecapacity24576; sortedwarp32opening andMAXDEPTH32; staticshared6144Bboundingbox/16384Bsummarization/3072Bforce. Allmodelsretain backendCUDAintrinsicdevicebodies; KA/JACC own9frameworkentrypoints+frameworkallocation, nohiddenCUDA.@cuda launch. Fivepredeclaredcases,46080completepublicstateoutputs and46080exactinputprefix values. All20captureindependentCPUtreecheck maxabs2.215981020903257e-6 withinfixed1e-5+1e-4relative; allbodymassbits preserved, totalmass1. Full100finaltreechecks and1300allocationguardchecks pass. RawtreeIDs notcrossoutputs.

## Reference audit

HistoricalCUDAjl direct_step_kernel! replaces nativeoctree/warpcollectiveBHapproximation withdenseallpairs andfusedpositionupdate, potentialcross-threadread/writehazard; no originalnumericalverifier. Preservedreference_history/main.jl. Coordinatorrestored8stagealgorithm+init andindependentCPUoctreeoracle beforetransfer; workercompletedguard/topology/APIhostreview andactualverification. No nativekernelchanges.

## Numerical and performance evidence

Configuration `bh_restored_eightstage_fivecases_fullstate_guards_topology_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T022534398516Z_bh/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Coordinatorinheriteddraft untestedbeforemapsreview. StaticatomicIncInt32APImappingbeforefirstprobe; noobservedcompilerfailure. WorkerfirstprogramhostfixturecopyR1 beforeauthoredJIT; secondprogram completephysicalstatePASS butsharedrecursivecheckerR1; thirdprogrampasses. OneauthoredCUDAjl9bodycandidate;3programchecks(twoauthoredJITprocesses),2sharedhostvalidationrepairs. KA/JACC1candidate/checkeach,9dispatchentrypointsconditionedonvalidatedsharedCUDAdevicebody. SharedT1 missingtimingdescriptor correctedonlybyhashboundall20-capturemetadatareview; noGPUseriesrerun/selectedtimes. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/bh-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific acceptance and implementation boundary

The independent CPU oracle constructs a spatial octree and follows sorted warp32 collective opening; its higher precision force/COM/integration is not a dense-force surrogate. All20 complete captures (46080 public state outputs each) satisfy the predeclared joint1e-5+1e-4*abs(reference) tolerance; max absolute error2.215981020903257e-6. Exact mass bits and totalmass1 are preserved. All100 final-tree DFS checks and1300 allocation-guard checks pass: no locked/duplicate/orphan cells, complete leaf partition, compact child lists, subtree count/start/sort consistency, conserved subtree masses and finite cell COM. These checks run outside ROI; raw allocator-dependent tree arrays are not retained as cross-model outputs. Momentum drift is recorded only as an auxiliary diagnostic, not an acceptance condition or claimed conservation theorem.

There are9 genuine kernel definitions (initializationonce plus8stagesperstep). CUDA.jl uses custom LLVM volatile loads/stores and therefore is classified hybrid_device_intrinsics under the study convention. KA/JACC use their own allocation and9 kernel dispatch entrypoints and call the validated device bodies with explicit CUDA block/warp/atomic/fence/shared-memory/rsqrt dependencies. They do not launch an undisclosed @cuda target, but are not claimed backend-independent device programs. This first faithful implementation retains CUDA static-shared arrays rather than claiming independent KA local-memory expressibility. Other-vendor support is untested, not unsupported.

## Shared failures and timing metadata

Coordinator-authored untested pipeline/oracle drafts were handed to maps_ports; original history and exact handoff hashes remain preserved. Worker host fixture-copy dispatch and recursive-checker captures were repaired separately from device translation. No authored GPU compiler rejection or device numerical repair occurred in those two shared failures. KA/JACC each passed their first candidate/check.

The inherited trial printer omitted timing_region even though the pre-GPU SCOPE, configuration and actual code define the same timer boundaries. Original result022149249567Z remains numerically verified but matched_performance_valid=false. Strict all-model reuse analysis022534398516Z adds a separate hash-bound effective-region annotation; all20 raw trial rows retain timing_region=None and all times/outputs/tolerances remain unchanged. No preferred new timing series was selected. The shared omission is T1 metadata failure. Exact review/evidence is in timing_region_metadata_review.json; derivedconfig is study_config_timing_metadata_review.json.

Native5component trial times are serialized at1microsecond resolution by std::to_string; total pipeline times retain their original precision. Full4model total-series means use only trials2–5. Existingfixture/expected_f64 binaries are required source-hash-bound assets generated by coordinator's unchanged C++initializer extraction and independent CPU oracle.
