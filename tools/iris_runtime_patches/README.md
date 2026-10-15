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

## HIP DMem physical allocation policies

`0002-dmem-hip-allocation-policies.patch` adds the HIP-specific API
`iris_dmem_set_hip_allocation_kind(mem, kind)`, set before any physical
allocation. Kind 0 is the unchanged device-memory default; kind 1 uses
`hipMallocManaged`; kind 2 uses coherent mapped `hipHostMalloc` with the matching
`hipHostFree`. DMem remains the owner in every case. The setter rejects regions,
USM-enabled objects, invalid kinds, and objects already allocated on a device.
The HIP deallocator also restores its owning context before releasing memory.

Apply patch 0002 after 0001 using `git apply --unidiff-zero --check <patch>`
followed by `git apply --unidiff-zero <patch>`, then rebuild as above.
This zero-context patch targets the exact reference revision; it avoids copying
upstream trailing whitespace as patch context. Review before applying elsewhere.
No complete IRIS sources are vendored here. The numeric mapped/coherent flag
values in the backend are the HIP ABI constants from ROCm 7.0.1; they avoid
expanding IRIS's minimal bundled HIP declarations.

`mallocFree-iris` forces physical allocation through `iris_mem_arch` but never
uses that pointer for a host GPU allocation/copy/launch. Untimed pointer-attribute
queries verify the requested physical kind. Distinct IRIS GPU-generated patterns
are written to all live buffers before full-array fetches, detecting aliasing
as well as stale contents. The bounded native/IRIS test covers all three kinds,
11 sizes (64 through 65536 bytes), eight simultaneous allocations per size,
786048 integer outputs in each model, and error-free release. Production
repeat/size defaults are retained and are not claimed as fully validated.
