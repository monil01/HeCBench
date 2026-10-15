# FP8 OCP E4M3 mathematical IRIS DMem port

Preserves original six shapes, repeat CLI/default100, warmup, uniform 1.0 E4M3
inputs, FP32 per-tensor input scales1, alpha1/beta0, and BF16 output. All device
allocations/data movement are DMem. HIP decode tasks apply FP32 input scales;
an IRIS bridge task calls real hipBLAS FP32 SGEMM using DMem pointers and the
IRIS-owned stream, followed by HIP BF16 rounding. No CPU fallback or direct
native benchmark allocation/copy/launch. This P1 expansion is NOT hardware FP8
GEMM: memory, accumulation precision, FAST_ACCUM behavior and performance differ.

`HECBENCH_IRIS_SHAPE=64,96,128 make run RUN_ARGS=2` bounds validation only.
`HECBENCH_IRIS_PATTERNED=1 HECBENCH_IRIS_SHAPE=17,29,256 make run RUN_ARGS=3`
exercises finite signed codes/subnormals and separate FP32 scalar input scales.
The extra fixture excludes E4M3 NaN encodings0x7f/0xff. Every output is checked
against independent double dot products at original rtol1e-2, plus BF16 exact
match counts. Production default sizes are retained, not claimed tested.

Native HIP source chooses FNUZ on gfx94x and OCP otherwise. This CUDA-semantic
port always uses OCP. MI100/gfx908 has no native FP8 matrix operation; native
build/run outcome is recorded separately, never claimed as a numerical PASS.
No native baseline timing or parity is invented on unsupported hardware.
