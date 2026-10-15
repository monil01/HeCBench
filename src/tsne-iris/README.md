# IRIS t-SNE

Preserves the complete original perplexity search, neighbor symmetrization,
Lagrange interpolation, forward/inverse GPU DFTs, attractive/repulsive forces,
and iterative updates. Original input files, CLI/defaults, output format,
coefficients and arithmetic kernels are retained under `native/`.

Persistent device vectors are IRIS DMem, including value initialization. Kernel
launches run through IRIS HIP tasks and the vendor-kernel API; genuine GPU Thrust
reductions/transforms/fills/minmax run on the IRIS-managed stream. Explicit
DMem transfers replace host/device vector copies and scalar iterator reads.
No CPU replacement for GPU arithmetic or shared runtime modifications.

P1: one visible HIP device, synchronous tasks, conservative hidden-DMem
dependencies and staged partial transfers. This prioritizes ownership and
correctness, not equivalent launch/overlap performance. Thrust's temporary
workspace remains vendor-managed; persistent benchmark vectors are DMem-owned.

`make run` uses the existing original `../tsne-hip/data` without copying data
into the source commit. `RUN_ARGS='-n 2 -d'` bounds iteration count without
changing the complete 60000-point input, grid/DFT sizes or algorithm stages.
Default 1000 iterations remains unchanged. Whole embedding outputs must be
compared, not timings or a small coordinate sample.

The original Gaussian point reader extracted into `points_host[i++]` while
testing EOF before extraction, permitting one out-of-bounds element at a trailing
newline. The port bounds writes and counts only successful extractions; native
CUDA/HIP reader repair and untouched-run comparison are recorded locally.
