# concurrentKernels — IRIS DMem/HIP

The original independent clock kernels run as asynchronous IRIS tasks on one
GPU. Each writes a distinct DMem region. An explicit fan-in dependency graph
joins all producers before the original 32-thread reduction and host fetch.
All producer outputs are checked against an independent exact integer formula.

Use an IRIS checkout with the DMem region ownership repair described in
[the runtime patch instructions](../../tools/iris_runtime_patches/README.md):

```sh
make IRIS_ROOT=/path/to/patched/iris
make run IRIS_ROOT=/path/to/patched/iris RUN_ARGS=4
HECBENCH_CONCURRENT_TRACE=1 ./main 4
```

The program enables `IRIS_ASYNC=1` and provisions `nkernels+1` IRIS streams.
The IRIS build-time asynchronous option is not necessary. HIP is queried only
for the original clock-rate input; IRIS manages all allocations, transfers,
execution, and synchronization. Scheduling and timing include IRIS overhead
and are not literal HIP stream/event API timings.

Opt-in trace mode records global GPU wall-clock intervals and requires producer
overlap when more than one producer is requested. Volatile diagnostic arithmetic
prevents compiler motion outside those intervals; trace timings are not
performance measurements. Uninstrumented tests preserve the source loop.
The reduction's final store is restricted to lane zero, eliminating the source's
same-address redundant writes without changing its arithmetic.

Validated on MI100/gfx908 with 1, 4, and 33 producers. Four-producer trace mode
demonstrated four overlapping producer pairs. The unchanged native HIP
four-producer baseline also passes.
