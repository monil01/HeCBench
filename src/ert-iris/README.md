# ERT IRIS
Preserves half2, float and double repeated operations, working-set/trial sweep,
32 MiB default, and gpu_blocks/gpu_threads arguments (run default 1024 256).
DMem.update(buf) refreshes each host-reinitialized working set; kernel accesses
are iris_rw and flushed before checksums. ERT_VALIDATION_BYTES optionally bounds
validation allocation (unset preserves original default). Host compiler hipcc
provides native half2 conversions; all device work is IRIS tasks, no HIP host
allocation/copy/launch calls. No CUDA backend supplied for HIP half2 implementation.
