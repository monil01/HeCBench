Validation on AMD HIP

Status: ok. Restored exact original BLAS/LAPACK DVC archive MD5 5dbd811ca4806146d3541791657806f1,size209183 into copied native baseline; archive paths verified. Native HIP and actual IRIS HIP backend3 agree exactly on CPU-reference maxError0.000244141<1e-3 for Ndim1/3,64 elements,polynomial degree7,100 repeats. Fixed IRIS reference comparison and restored both default run cases. Native sibling untouched; copied baseline build uses original archive and installed gfortran runtime path. Default8000-element workloads not executed; recorded native build retries total4.89s.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/axhelm-iris/attempts/attempt_05` (local, not committed).
