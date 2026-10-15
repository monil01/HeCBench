# Patch04: HIP stream lifecycle and matching async allocation/free

Incremental dependency: apply patches0001,0002, then
`0003-explicit-hip-stream-control.patch` (verified coordinator commit49b14e1f5)
before this patch. This patch was generated against an isolated source copy of
the coordinator runtime including that exact03 implementation. No live runtime
or previously committed patch was edited.

```
patch -l -p1 < /path/to/HeCBench/tools/iris_runtime_patches/0004-stream-lifecycle-allocation.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=OFF \
  -DUSE_SIGHANDLER=OFF -DIRIS_ASYNC_STREAMING=ON
cmake --build build -j8
```

Public C APIs:
The patch uses zero context to avoid inherited trailing whitespace in source
comments. Use the shown `patch -l` command: loose whitespace matching is needed
for one removed upstream comment. Plain `git apply --unidiff-zero` is not the
application command for this patch. Reapplication was checked byte-for-byte
against the tested isolated runtime sources.

Public C APIs:

* `iris_set_malloc_async(0|1)` waits for submitted work before changing the
  existing Platform malloc policy. Initial policy/defaults are unchanged.
* `iris_hip_stream_create(device,index,nonblocking)` creates a physical stream
  in an **empty** runtime-owned slot. Priority0 and flag0/1 preserve ordinary
  HIP stream creation behavior. Occupied slots and invalid controls fail.
* `iris_hip_stream_destroy(device,index)` waits for submitted work and the
  physical stream, destroys it, and clears both slot aliases. It rejects a
  slot with live async allocations. Never submit a task to an empty slot.
* `iris_hip_lifecycle_stats(device,uint64_t[8],reset)` reads/resets counters:
  explicit-slot creates, destroys, device async allocations/frees, legacy
  device allocations/frees, successful physical H2D calls and bytes. Initial
  stream creation, priority reconfiguration, managed/pinned allocations and
  USM/no-op transfers are deliberately not counted in those categories.

Control calls are quiescent operations, not a concurrent stream-management
protocol: callers must serialize them against new task submissions and host
memory destruction. Existing03 task binding remains unchanged. The03
`iris_dev_get_stream` return value is a raw HIP stream handle, not a pointer to
a handle. Pool attributes are configured/query-checked by ordinary HIP pool
property APIs; allocation/copy/kernel ownership remains entirely IRIS DMem.

Async frees use a per-device pointer-to-allocation-stream map protected by a
mutex. Only successful kind0 hipMallocAsync calls enter the map; frees use
hipFreeAsync on that recorded stream even if the global malloc policy or DMem
recommended stream subsequently changes. Managed kind1 still uses hipFree;
coherent pinned kind2 still uses hipHostFree. Ordinary kind0 hipMalloc stays
hipFree. All counters increment only after successful physical dispatches.

Validation (AMD MI100, HIP-only, isolated source/build):

```
make -C src/streamOrderedAllocation-iris IRIS_ROOT=/path/to/patched/iris
make -C src/streamOrderedAllocation-iris runtime_test IRIS_ROOT=/path/to/patched/iris
cd src/streamOrderedAllocation-iris
IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=4 ./runtime_test
HECBENCH_SOA_ELEMENTS=4099 HECBENCH_SOA_REPEATS=2 \
  IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=4 ./main
```

Runtime regression source lives alongside the consuming benchmark (separate
benchmark commit). All259 outputs pass for device/managed/pinned allocation
kinds; physical pointer attributes match. Explicit create/destroy counts1/1,
H2D count2, and async allocation/free counts3/3 (device) or2/2 (managed/pinned
output plus two device inputs) remain correct after policy-off **before free**.
Invalid controls, duplicate creates, double destroys and destruction with live
async allocations are rejected without poisoning the runtime error count.
The six original allocation-policy/check/timing phases at4099 elements and two
iterations pass all outputs and exact physical alloc/free/H2D counts. Actual
queried pool release thresholds are0 and18446744073709551615 (ULONG_MAX).
Task synchronization adds IRIS scheduling overhead; no performance equivalence
to direct HIP enqueue overhead is claimed. Historical/local full evidence is
under `.porting-state/porting_logs/streamOrderedAllocation-iris/attempts/attempt_02`.

Base SHA256 fingerprints (after03, before04):

```
include/iris/iris_runtime.h 99d554f6c7bbb6710401296082820310fb6a7a1d9d286167a7d6099f22d983b4
src/runtime/CAPI.cpp 9d76addca74025adc0f5e958f7e353fa9451b5a7ff10b080501e10cab241a145
src/runtime/DeviceHIP.h 7a6d3067bf5e73571e9d3af657f53a35bc2e785dfff92fbe65359c42ea361e1d
src/runtime/DeviceHIP.cpp 45ae4e2e7e2525cc23bb5b4a2eeee2c96247d521c6e1aeea49bfd0de41ab751f
```
