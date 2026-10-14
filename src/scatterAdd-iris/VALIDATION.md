Validation on AMD HIP

Status: ok. Native HIP and actual IRIS HIP backend3 PASS for16385 items/vector_dim8 and8193/vector_dim64,output32,default1000 repeats,seed2,10 warmups. All256/2048 output floats match exactly and remain finite after warmup,after original kernel1,and after original kernel2; phase totals equal original reference times1/1001/2001. Original two shared-memory algorithms restored; static capacities1024/4096/16384 words adapt IRIS dynamic-shared launch limitation(P1,possible occupancy changes). Default10-million-item cases and prior native719 failures not revalidated; defaults preserved. Native sibling untouched.

Run with IRIS_ARCHS=HIP from this directory. Full source defaults are preserved; bounded cases are validation coverage only. Attempt history and commands: `.porting-state/porting_logs/scatterAdd-iris/attempts/attempt_02` (local, not committed).
