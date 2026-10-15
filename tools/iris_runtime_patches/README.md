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

## Explicit HIP task streams and priorities

`0003-explicit-hip-stream-control.patch` builds on 0001/0002. Apply with
`git apply --unidiff-zero`, then rebuild. It provides:

- `iris_task_set_stream_index(task, index)`: bind an unsubmitted task to an
  existing runtime-owned stream, using same-stream task transfer policy.
- `iris_hip_set_stream_priority(device, index, priority)`: recreate a runtime
  HIP stream with a requested nonblocking priority before any work uses it.
  Call only while quiescent; this is not a concurrent stream migration API.
- HIP's implementation of the existing `iris_dev_get_stream` query, returning
  the borrowed stream handle for metadata inspection. IRIS retains ownership.

Invalid devices/indices and post-submission stream binding are rejected.
Reconfiguration synchronizes/destroys the old stream, maintains the backend's
stream aliases, and does not alter DMem ownership or task kernel arithmetic.

`streamPriority-iris` validates actual queried priorities on MI100: baseline 0,
training -1, and prefetch +1. It preserves training-stream order and the
forward-done/prefetch-done cross-stream graph. Odd 4099-element/3-batch and
65536-element/5-batch fixtures compare every activation, gradient, updated weight,
and prefetch value with native HIP, byte-exact in both modes. CPU source-derived
reference checks also pass. No preemption or production-default speedup claim
is made; task/data-management overhead differs from literal native stream APIs.

## Stream lifecycle, peers, and concurrent system atomics

The full catalogue includes ports requiring these additional, narrow extensions:

* [0004](0004-stream-lifecycle-allocation.md): real stream create/destroy,
  borrowed legacy-default stream slots, stream-ordered allocation/free, and
  physical lifecycle counters. Consumers: streamOrderedAllocation and
  streamCreateCopyDestroy.
* [0005](0005-peer-dmem.md): genuine whole-DMem HIP peer copies and read-only
  remote-GPU kernel bindings, with no silent input migration. Consumer: p2p.
* [0006](0006-system-atomic-dmem.md): DMem-owned coherent host storage with a
  borrowed CPU view for real simultaneous CPU/GPU system-scope atomics.
  Consumer: atomicSystemWide. Coherent host allocation and equivalent atomic
  CAS operations are explicit platform adaptations, not VRAM atomicity claims.

Use the exact stock reference revision above in a separate private tree. Apply
the final committed patches in order (substitute absolute paths):

```sh
git apply /path/to/0001-dmem-region-ownership.patch
git apply --unidiff-zero /path/to/0002-dmem-hip-allocation-policies.patch
git apply --unidiff-zero /path/to/0003-explicit-hip-stream-control.patch
patch -l -p1 < /path/to/0004-stream-lifecycle-allocation.patch
patch -l -p1 < /path/to/0005-peer-dmem.patch
git apply --unidiff-zero /path/to/0006-system-atomic-dmem.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=OFF \
  -DUSE_SIGHANDLER=OFF -DIRIS_ASYNC_STREAMING=ON
cmake --build build -j8
```

Patches04/05 intentionally use whitespace-tolerant `patch`, not Git's
whitespace option. Patch06 is rebased against the complete01–05 baseline;
do not skip05 when applying the current06. Read each patch document for API
preconditions, environment settings, tests, and qualifications. Caller arrays
remain DMem-owned. `IRIS_ARCHS=HIP` selects actual AMD GPU execution; no CPU
backend is a benchmark-completion path. Set `IRIS_ROOT`/`IRIS_BUILD` in each
benchmark's build, rather than overwriting a runtime used by another process.

Vendor-library workloads use [the host2hip bridge](../iris_vendor_bridge.md)
to operate on DMem-resolved GPU pointers and the runtime-owned stream; the
bridge is not a CPU substitute. Hardware-unavailable low-precision operations
have documented mathematical GPU equivalents, not native low-precision
performance/parity claims. Per-port documentation and local validation
manifests retain bounded-workload, baseline-repair, and scientific-reference
qualifications. Directory or committed-layout counts alone are not proof of
numerical correctness; inspect `python3 tools/iris_status.py --json` together
with the corresponding local evidence.
