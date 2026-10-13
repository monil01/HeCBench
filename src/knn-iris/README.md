IRIS data-memory kNN port with unchanged 4096 reference/query points, dimension
68, k=20, seed2, repeat argument, default repeat100, and precision/index checks.
Three original HIP kernels run as synchronous IRIS HIP tasks on one AMD GPU.

The benchmark-local DMem helper registers host-backed buffers with
`iris_data_mem_create`, updates host-modified buffers explicitly, preserves
mutable intermediates with `iris_rw`, passes pointer offsets via the IRIS task
API, and flushes outputs. Native allocation, copying, and launches are absent.
The host uses HIP types/toolchain for source layout compatibility.

IRIS is finalized before main returns to avoid late static-destructor ordering
with runtime history pools. Submitted non-permanent tasks follow IRIS automatic
release semantics. Full failed attempts and fixes are preserved in local state.
Validation used repeat1 at the unchanged problem size and matched both native
accuracy summaries. Default100 is retained but was not validated in this pass.
