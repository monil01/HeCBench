Validation on AMD HIP

Status: ok. Native HIP DEBUG and actual IRIS HIP backend3 complete256 bodies/2 timesteps,512/3,and256/100. All9 printed scalars per body are finite and match at rtol0.005/atol1e-7 (2304,4608,2304 scalars), including the original100 timestep count. Original stage blocks,256 threads,tree traversal,seed7 and default100000 bodies/100 timesteps retained. Bounded body-count coverage accepted; full100000-body workload not validated and prior timeouts preserved. Native sibling untouched.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/bh-iris/attempts/attempt_03` (local, not committed).
