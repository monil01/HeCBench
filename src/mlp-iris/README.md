# MLP with IRIS DMem HIP

Complete FP32 forward propagation: every layer performs W-transpose times
input, optional bias, and optional ReLU, including the final layer. CLI,
seed 123, uniform[-1,1] inputs/weights/bias generation order, 300 warmups,
repeat count, and all four original make-run defaults are retained.

P1: one-output-per-thread HIP kernels replace fused hipBLASLt matmul
epilogues. DMem owns all buffers and IRIS tasks launch every GPU operation.
There is no vendor-heuristic or performance-equivalence claim. The unchanged
CUDA reference.h computes an independent complete forward result; every
final output is checked at the source absolute tolerance 1e-2, with additional
finite checks. Native numerical diagnostics only add complete output printing
to an isolated source copy and do not replace any native BLAS operations.

IRIS_MLP_DUMP=1 prints all output values. Bounded three-layer cases exercise
all four bias/ReLU combinations and a separate one-layer odd-size case.
Default 784/512 sizes are retained, not claimed as performance-tested.
