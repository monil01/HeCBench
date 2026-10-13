Nine managed CUDA inputs/outputs become host-backed IRIS dmem without native
allocation/copies in the host. Reuses native HIP wave reductions and source
struct layouts. No-shift warmup and accumulation, full input reset via update,
and shift accumulation are preserved. Exact scalar ABI is int,float,float,
float,bool (one-byte bool). Default repeat=10000 and full3199-block grid remain.
HECBENCH_ARTIFACT dumps every force, shift force and cluster-list entry after
both phases for full native comparison. Isolated native copy adds only dumps;
validation uses repeat3 and source's full atom/grid sizes.
