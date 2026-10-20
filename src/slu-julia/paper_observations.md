# SLU — verified conditional full-source sparse factorization

RQ1: full Sandia/Xyce ASIC100k, add32 and two inactive-pivot controls. All20 complete artifacts pass exactactualarrayinputs, tolerantfullLU/X dual references and independent full sparse product/original-system backward-error gates. The original checksum checker accepts an executed CPU counterexample; that is not an observed GPU failure. Sourceordering repaired after13→0 matchedracewarnings. No forwardconditioning/circuitphysics guarantee.

RQ2: onecompletecandidate/oneactualdevice-JITprogram pertarget, sixentrydefinitions/twohelpers and48,036selecteddispatches includingfullwarmup+5. Zero targetcompiler failures. Actualscopedtokens in effort_summary.json preserve cachedcontext, native/commonwork and overlappingverification; no exclusivebilled/coldstarttarget ranking. Ownfixture/adapterfailures retained separately.

RQ3: mean trials2–5 native/CUDAjl/KA/JACC seconds: 0.725321232, 0.793839387, 0.828081383, 0.885247403. Original fullnumericGPUlevelloop+CPUlaunch/CABI/control/contextcompletion; sourceCPUpreprocess/symbolic/solve outsideROI. Allfivetimes retained, nooutlierdrop. Conditionalnative excludeduntouchedaggregates.

RQ4: CUDA source warp/atomic/shared/thread semantics are backendextensions inalltargets. KA preserves8/16/128reservedsharedbytes withValstaticstorage; CUDA/JACCdynamicallocationretained. Full4.113GBscratch preserved andreadbackoutsideROI. No profiler/counter/roofline or puredevicecausalityclaim.

RQ5: feasible withCUDA-specifichybridinteroperation; source-warp algorithm here isnotcertifiedbackend-independentKA/JACC. Noframeworkunsupportedclaim.
