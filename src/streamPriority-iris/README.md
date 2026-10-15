# streamPriority — IRIS DMem/HIP

The original forward/backward/SGD/prefetch kernels run as IRIS tasks, with DMem
for every tensor. Priority mode uses actual IRIS-owned HIP streams: training at
priority -1 and background prefetch at +1 on MI100. The ordinary baseline uses
an IRIS-owned priority-0 stream, not the literal native null-stream handle.

The task graph preserves training-stream order, forward-done → prefetch, and
prefetch-done → next-forward dependencies. Prefetch may overlap backward/update;
this is not a sequential replacement with priority labels. Actual stream
priorities are queried and validated; invalid runtime API indices are tested.

Apply patches 0001–0003 from
[the runtime patch instructions](../../tools/iris_runtime_patches/README.md):

```sh
make IRIS_ROOT=/path/to/patched/iris
make run IRIS_ROOT=/path/to/patched/iris RUN_ARGS=300
```

The original `<repeat>` CLI, 16,777,216-element default, four kernel bodies,
initial values, and learning rate remain unchanged. Optional compile-time
`HECBENCH_STREAM_PRIORITY_N` bounds validation in both native references and IRIS:

```sh
make -B IRIS_ROOT=/path/to/patched/iris EXTRA_CXXFLAGS=-DHECBENCH_STREAM_PRIORITY_N=4099
OMP_NUM_THREADS=2 IRIS_ARCHS=HIP ./main 3
```

Every activation, gradient, updated weight, and prefetch value is checked against
the unchanged source-derived CPU oracle at the original absolute tolerance 1e-3.
Native optional observation hooks dump all four arrays after each scheduling mode,
outside its timing region. Set `HECBENCH_STREAM_PRIORITY_DUMP` to a unique filename
prefix; native dumps additionally require `HECBENCH_STREAM_PRIORITY_VALIDATE`.
`verify.py` checks complete finite native/IRIS arrays byte-for-byte.

Odd 4099-element/3-batch and 65536-element/1- and 5-batch fixtures passed in both
modes, all 1081368 output floats per model byte-exact. CPU maximum errors were below
1.2e-7. The full default size/300-batch workload was not run. IRIS scheduling,
transfer, and callback overhead differs from literal HIP stream/event timings;
no production speedup or guaranteed GPU preemption claim is made.
