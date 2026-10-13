BGMV uses IRIS DMem for all device storage and IRIS tasks for HIP kernel
launches and output flushes. Build with `make`; run with `IRIS_ARCHS=HIP make
run`. Override `IRIS_ROOT`, `IRIS_LIBDIR`, `CXX`, `HIPCC`, or `HIPFLAGS` for
other installations. `RUN_ARGS` selects a single argument configuration;
otherwise `make run` preserves the eight CUDA Makefile configurations.

The CUDA defaults, seed 19937, six block sizes, fp16 conversion, fp32 CPU
reference, and absolute error threshold 0.1 are preserved. Shrink accumulates
each repetition; expand honors `--add_to_output`. As in CUDA, `--vectorize`
affects shrink only, and expand does not multiply by `--scaling` (the original
reference call converts scaling to bool on a zero reference array).

The original grid becomes IRIS work sizes `(tokens * block_size, columns)`
with local sizes `(block_size, 1)`. Each exported kernel retains the baseline
hipCUB reduction and scalar/vector loop. The optional CUDA kernel uses CUB
and can be compiled with `make cuda`; it is not the AMD validation backend.
Vector loads use float4 as in the baseline. Rows should have hidden sizes
divisible by eight for aligned vector loads; scalar kernels support arbitrary
positive hidden sizes.

Timing includes IRIS task creation/submission and synchronous completion;
it is not directly comparable to native HIP asynchronous launch timing.
Numerical failures return nonzero while retaining the original printed check.
The host reports available backend/device identities and selects an AMD HIP
device explicitly; absence of one is a blocked run, never a passing fallback.
