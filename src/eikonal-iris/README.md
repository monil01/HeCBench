Eikonal FIM IRIS dmem port
=========================

Preserves the source FIM active-list and ping-pong solver/reduction/neighbor tasks, 4x4x4 workgroups, double fields, uint active counts, int dimensions/iterations, default 256 cubed volume, Makefile 512 cubed run, both map modes, seed voxel, tolerance and full NRRD artifact output. Host opaque buffer handles own iris::DMem plus host storage. Partial upload/reset uses update; all partial kernel writes use rw and required host results flush out. No native GPU allocation/copy APIs in host.

Validation runs unchanged native HIP sources in an isolated snapshot (CUDA shared timer/main/header dependencies also snapshotted), using existing CLI -s 32 -m 0/1 -i 10. Every NRRD double and convergence/block verdict is compared; all distances must be finite, seed zero, and atol=rtol=1e-12. Native siblings unchanged. Padding beyond dimensions has a pre-existing source map indexing issue, so test dimensions are multiples of BLOCK_LENGTH.
