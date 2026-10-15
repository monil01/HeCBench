# Intrinsic casts through IRIS DMem

The HIP backend preserves both original intrinsic kernels and CLI (`n repeat`).
All device storage and dispatch use IRIS DMem/tasks. `HECBENCH_DUMP=path` writes
both complete int64 output vectors, first cast-from-FP then cast-to-FP.

The original CUDA/HIP initialization allocated n elements but wrote indices
1..n, leaving index zero undefined and corrupting the heap at index n. The
documented baseline repair, applied identically to CUDA/HIP/IRIS, stores the
intended sequence `22.44 / (i+1)` and original integer constant at indices 0..n-1.
No intrinsic or launch geometry changed. Old sources/attempts remain in local
porting evidence. All outputs match the repaired native HIP baseline byte for
byte for n=1, 257 and 1000000; cast-from-FP additionally matches an independent
CPU bit-pattern and four-rounding-mode oracle for every element. The second
kernel is validated against native HIP, not an independent CPU emulation.
