f16sp IRIS DMem port
===================

All three custom CUDA dot products retain their arithmetic, five grid sizes,
128-thread blocks and packed binary16 input. Kernel3 follows the CUDA scalar
CUB block sum of x+y; native HIP uses a float2 block sum. Both are checked.
The library operations are complete two-stage HIP GPU implementations launched
through IRIS: FP32 accumulation with FP16 result for BLAS DotEx, and rounded
half2 multiplication followed by FP32 reduction for CUB DeviceReduce::Sum.
They are not vendor-library performance substitutes. Tree ordering differs;
general inputs can have normal FP32 accumulation-order differences.

No native HIP allocation or launch occurs in the IRIS host. All arrays, partial
sums and scalar results are IRIS DMem, and results are explicitly flushed.
Warmups clear scalar accumulators, unlike native's initially undefined warmup
accumulator; timed native already clears it. This adds an IRIS task per custom
kernel invocation, and timings include task/flush overhead.

`make run` preserves repeat1000, 1048576 blocks and warmup1000. Override matching
native/IRIS compile flags with EXTRA_CFLAGS for bounded validation, for example
`-DNUM_OF_BLOCKS=1024 -DF16SP_WARMUP=2`. Input formula and all five grid ratios
remain unchanged. Native sibling changes only expose those optional controls,
line-buffer diagnostics, and F16SP_VALIDATE exact hexadecimal scalar reports.
There is no claimed native algorithm repair or explanation of the historical
signal11 until reproduced with diagnostic output. Historical attempts remain.
