# Stream-ordered DMem allocation

Requires runtime patches0001..0004, compiled with asynchronous support. Apply
04 with `patch -l -p1` as documented in
`tools/iris_runtime_patches/0004-stream-lifecycle-allocation.md`; select that
isolated runtime with `make IRIS_ROOT=/path/to/patched/iris`.

Preserves all three policies in both checking and timing phases: synchronous
device allocation/free, async allocation/free with pool release threshold0,
and post-sync async allocation/free with thresholdULONG_MAX. Every device
buffer is an IRIS DMem; transfers and vector-add launches are IRIS tasks.
HIP calls in the host configure/query pool properties or synchronize the
borrowed runtime-owned stream, never allocate/copy/launch device work.

Runtime04 records the actual allocation stream and uses matching hipFreeAsync
for async kind0 memory; legacy, managed and pinned free paths stay distinct.
Per-phase backend counters assert exactly3 allocations and frees plus2 physical
H2D copies per iteration. Pool thresholds are queried after configuration;
numeric outputs are checked in every phase, including timed phases.

Defaults remain33554432 elements,100 iterations and the original rand() inputs.
`HECBENCH_SOA_ELEMENTS`/`HECBENCH_SOA_REPEATS` bound validation only, identically
in native CUDA/HIP. `HECBENCH_SOA_DUMP=prefix` writes all six complete results.
Bounded native/IRIS checks cover odd4099/2,65537/3 and8192/7 element/repeat
pairs, all exact outputs and independent CPU rand/float32-add references.
Production-size defaults are not claimed as tested.

`make runtime_test` and `IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=4 ./runtime_test`
test actual stream lifetime, invalid/live-slot controls, pointer kinds and
matching frees even after global policy-off. All259 outputs pass for device,
managed and pinned cases. No CPU fallback or plain-Mem device storage is used.

Performance qualification: IRIS scheduling and synchronous task completion add
overhead relative to direct native enqueue; physical allocation/free policies,
stream ownership, transfer counts and pool controls are preserved, not identical
native overlap or speed. Original license is retained in LICENSE;
untouched snapshots and previous failed/deferred attempts remain local.
