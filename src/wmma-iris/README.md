# WMMA IRIS dmem port

Host allocation and submission use IRIS DMem/tasks exclusively. HIP kernel entrypoints retain the native rocWMMA fragments and both source implementations; rocWMMA headers must be installed. HIP currently requires wave64 and probes it on device.

Arguments retain CUDA order: implementation M N K repeat verify. CUDA 16x16x16 fragments, all input formulas, alpha0.5/beta2, thirty warmups, verification tolerance and timed repeats are retained. Native HIP uses16x16x32 fragments; compared common K multiples32 produce the same GEMM. CUDA supports K multiples16, retained here. Original truncated64x64 implementation1 grid behavior is unchanged for non-full tile shapes.

Configure CXX/HIPCC/IRIS_ROOT/IRIS_BUILD; run with IRIS_ARCHS=HIP. RUN_ARGS optionally selects a case; default run suite retains CUDA/HIP source arguments. Validation limitations and evidence are in the local manifest. CUDA backend is not provided or validated. Timing includes synchronous IRIS task overhead rather than bare HIP queue submission time.
