# IRIS runtime fixes used by benchmark ports

These are narrow source patches, not a vendored IRIS runtime. The reference
revision is ORNL/iris `ff57e620a92fcf62eff3568d9c6eeaaf078f0340`.
Keep a separate checkout/build when validating alongside existing ports.

## DMem region ownership

`0001-dmem-region-ownership.patch` prevents `DataMem::~DataMem()` from calling
the backend deallocator for borrowed subregion addresses. The parent alone owns
the allocation. Without this patch, releasing a DMem with materialized regions
causes HIP invalid-pointer errors, including a double free of region zero.
The patch does not change task scheduling or arithmetic.

From a separate IRIS checkout, substituting the actual HeCBench path:

```sh
git apply --check /path/to/HeCBench/tools/iris_runtime_patches/0001-dmem-region-ownership.patch
git apply /path/to/HeCBench/tools/iris_runtime_patches/0001-dmem-region-ownership.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Then pass `IRIS_ROOT=/path/to/patched/iris` to the benchmark's `make` commands.
`concurrentKernels-iris` enables asynchronous execution through `IRIS_ASYNC=1`;
the compile-time `IRIS_ASYNC_STREAMING` option is not required.

Regression: run `concurrentKernels-iris` with 1, 4, and 33 producers, check all
producer/reduction outputs and successful teardown, then use
`HECBENCH_CONCURRENT_TRACE=1` for an opt-in GPU wall-clock interval test.
Trace instrumentation changes timing and must not be used for performance
comparisons. A passing scalar alone is not evidence of concurrent execution.
