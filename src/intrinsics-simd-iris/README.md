# Packed SIMD intrinsics with IRIS DMem

Original CUDA input pattern, 118-operation sequence/XOR, element/repeat CLI,
256-thread geometry and default1000000/100 retained. HIP device-only lowering
extracts 8/16/32-bit lanes and explicitly handles signedness, modulo additions,
saturation, comparisons, predicates, SAD and ReLU. No opaque library operation.
CUDA kernel retains original intrinsic calls. Scalars remain32bit, buffers DMem.

No native HIP sibling exists: external native_status is
blocked:no-native-hip-baseline and native compile/runtime timings are null.
As user authorized, actual MI100 HIP execution is checked in-program against
an independent widened/modulo host model derived from CUDA semantics, not a
fabricated native baseline. Every output word compared, not just checksum.
Additionally all118 individual intrinsic outputs/predicates checked over every
byte pair, all19^3 boundary triples and4096 deterministic random triples to
prevent XOR cancellation hiding errors. Hard-coded host model anchors cover
negative average rounding, unsigned saturation and wrap-before-max.

Semantics inspected in official NVIDIA CUDA12.4 compiler header archive
(device_functions.hpp, kept only in exclusive scratch, not vendored) and
[NVIDIA SIMD API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)
and [PTX SIMD rounding](https://docs.nvidia.com/cuda/parallel-thread-execution/#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2).
Signed average rounds odd sums away from zero; fused add min/max wraps each
lane before comparison. Original CUDA intrinsic names/order are retained in
HIP wrappers. Installed/downloaded headers are reference only, not execution.
