# Batched SDDMM through IRIS DMem and genuine hipSPARSE

Retains original seven arguments, all batches, inputs/seeds, alpha=1/beta=0,
no-transpose row-major operation, 0.01 full CPU oracle and both default cases.
Inputs are generated before IRIS initialization. Uses the unchanged native HIP
workaround: separate descriptors per batch and a per-batch SDDMM loop because
AMD hipSPARSE does not support the CUDA strided-batch SDDMM descriptor path.

The committed IRIS vendor bridge resolves DMem-owned GPU inputs/output/workspace
and IRIS-owned stream into the original hipSPARSE operations. No native allocation,
copy, or CPU arithmetic replaces the workload. Setup queries workspace size, then
a second task executes preprocessing and the full repeat/batch loop. All vendor
errors abort via bridge; explicit stream synchronization protects task lifetime.
P1: task/setup overhead differs from native; no performance parity claimed.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS="3 33 29 17 40 3 1"`.
Optional HECBENCH_DUMP_PREFIX exports every batch output. Bounded cases are
validated; production defaults and a CUDA backend are not. Original notice is in
NOTICE. Original CUDA/HIP sources remain untouched.
