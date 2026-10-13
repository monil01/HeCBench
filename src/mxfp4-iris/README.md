# MXFP4 IRIS dmem port

The CUDA software FP16/BF16-to-FP4 quantize/dequantize algorithm is retained in HIP kernels with named IRIS entrypoints. Host arrays are owned through IRIS DMem; no native device allocation, copies, or launches are used by the host.

Build with `make`, and run with `IRIS_ARCHS=HIP ./main <numel> <repeat>`. Both arguments are mandatory, as in CUDA. Positive element counts must be multiples of 64. The run target retains the original 536870912-element, 100-repeat defaults. CXX, HIPCC, IRIS_ROOT and IRIS_BUILD are configurable.

Original seed123 and correctness checks are preserved for both dtypes. Optional HECBENCH_DUMP_PREFIX writes exact output words for observational comparisons; it does not alter inputs or computation. Native HIP baseline sources are untouched.

Validated on AMD MI100 HIP backend with 64, 192 and 4096 elements, each with three iterations: native PASS verdicts and 8704 exact output-word matches including negative and finite values. The production-sized default was not executed. CUDA backend is not included or validated. Invalid arguments are rejected rather than retaining the original uninitialized-output behavior. Tasks submit synchronously, so reported wall-time bandwidth includes IRIS overhead and is not kernel-only native performance.
