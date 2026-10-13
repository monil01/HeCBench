IRIS DMem softmax Jacobian preserves dimensions128,512,2048,8192, block sizes64..1024,
batch/repeat arguments, seed123 and 1e-3 CPU reference tolerance for both kernels.
HIP kernel uses the native HIP sibling's wavefront-size-agnostic reduction, preserving
CUDA mathematical semantics. Static aligned shared storage covers max8192+1024 float
entries in each specialization; HIP task ABI has no dynamic-shared launch parameter.
Conservative iris_rw intermediates, explicit DMem host updates and output flushes.
Timing includes IRIS overhead. CUDA source supplied but not validated.
