# CSR-to-CSC conversion through IRIS DMem

Despite its name, the source benchmark performs numeric CSR-to-CSC conversion,
not sparse addition. The full output offsets, row indices and values are retained.
The original command line, input helpers, seeds, repeat count and 0.01 check are
preserved. Inputs are generated before IRIS initialization to preserve RNG state.

P1: the authorized vendor-operation replacement is a single HIP thread doing
histogram, prefix sum, and stable scatter. It preserves the mathematical operation
and sorted output, but does not reproduce cuSPARSE/hipSPARSE scheduling or speed.
Device storage and dispatch use only IRIS DMem/tasks, never host CPU conversion.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS="33 29 40 3 1"`.
The default workload is unchanged and has not been benchmarked. With verification
enabled, `HECBENCH_DUMP_PREFIX` exports every output for observational comparisons.
Native CUDA/HIP sources are untouched; validation uses a dump-only native copy
and the original dense-transpose CPU oracle. HIP backend only is validated.
