# mallocFree — physical IRIS DMem allocation policies

This port times physical managed, device, and coherent mapped-host allocations,
not just lazy DMem object construction. All GPU allocations belong to DMem;
the HIP APIs are selected inside the runtime backend. The borrowed native
pointer is used only for pointer-attribute queries, never host GPU allocation,
copy, or kernel launch. Numerical work is an IRIS HIP task.

Apply runtime patches 0001 and 0002 from
[the runtime patch directory](../../tools/iris_runtime_patches/README.md), then:

```sh
make IRIS_ROOT=/path/to/patched/iris
make run IRIS_ROOT=/path/to/patched/iris RUN_ARGS=536870912
```

The original `<total global memory size in bytes>` CLI, 19 size classes, and
500 live allocations per class are retained. Allocation policy/order and
physical lifetime are preserved, although timings include IRIS object/API
overhead. An insufficient budget is rejected instead of indexing `size[-1]`.

Outside the timed allocation/free intervals, every live allocation receives a
different GPU-generated integer pattern. All allocations are filled before
fetching any buffer, so aliasing is detectable. Every element is checked on the
host; HIP pointer attributes also verify each requested physical allocation kind.
This extra diagnostic workload is not part of the original timing region.

Bounded validation can override the compile-time repeat count, without changing
its production default:

```sh
make -B IRIS_ROOT=/path/to/patched/iris EXTRA_CXXFLAGS=-DNUM_ITER=8
IRIS_ARCHS=HIP ./main 1048576
make -B -C ../mallocFree-hip EXTRA_CFLAGS='-DNUM_ITER=8 -DHECBENCH_MALLOC_VERIFY'
../mallocFree-hip/main 1048576
```

The references gain opt-in numerical diagnostics and a guarded `NUM_ITER` macro.
The HIP reference's mapped-host deallocator is repaired to `hipHostFree`,
matching the already-correct CUDA `cudaFreeHost` path. Native diagnostics use
the same full-array pattern oracle and physical-kind checks. Production default
size/repeat coverage must not be inferred from bounded tests.
