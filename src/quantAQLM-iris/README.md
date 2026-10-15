# AQLM matvec with IRIS DMem

Preserves the full original batch of four 4096-input/12288-output matvecs,
three 65536x8 FP16 codebooks, seed 123 and code generation. All allocations
and launches are IRIS DMem/tasks with the HIP wave64 kernel on this system.

Baseline defect repaired identically in CUDA/HIP/IRIS: output size and batch
stride were divided by 32 although the kernel emits one half for each of
12288 rows. Allocate four complete vectors and use stride 12288. No matrix,
input/codebook, arithmetic, or reduction was changed. Untouched source and
old candidates are retained in the local attempt evidence.

`HECBENCH_DUMP=path` writes all 49152 FP16 outputs. Set `HECBENCH_VERIFY=1`
for an independent CPU decoded-matrix dot-product check of every output,
with a double-precision sum and tolerance atol=0.01,rtol=0.002 to account for
the GPU's intermediate half-precision fused accumulation. This diagnostic is
outside the original kernel timers and not part of normal benchmark timing.
The full original workload passes this oracle in native HIP and IRIS; all
native/IRIS output half bit patterns match exactly. CUDA execution was not
available; kernel.cu remains unvalidated on NVIDIA hardware.
