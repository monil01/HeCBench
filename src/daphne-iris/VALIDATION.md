Validation on AMD HIP

Status: deferred. IRIS actual MI100 HIP backend3 passes current CUDA-derived deterministic projection reference at100000 points for default-p1 and-p3 batches: max errors3.662e-4,3.662e-4,4.883e-4,validity mismatches0,all finite. Existing HIP sibling is incompatible legacy data-driven points2image; build fails missing src/daphne-cuda/src/common/main.cpp (also benchmark.h/datatypes.h and legacy data absent). Native baseline repair/replacement not authorized; no compatible external native verification or native runtime. Current CUDA synthetic inputs,LCG seed,batch/default semantics retained; finite/task-status checks strengthened.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/daphne-iris/attempts/attempt_03` (local, not committed).
