Validation on AMD HIP

Status: ok. Native HIP and IRIS_ARCHS=HIP each passed all 36 original-reference checks at 8 rows, n=256,512,1024,2048,4096,8192; float/half/bfloat16 and versions 1/2, 100 warmups and repeat=1. Original native device arithmetic and default rows=4096 retained. Native wall runtime unavailable for first measurement because it included lock wait; IRIS wall runtime 6.18s excludes wait. Full 4096-row workload not executed.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/addBiasResidualLayerNorm-iris/attempts/attempt_04` (local, not committed).
