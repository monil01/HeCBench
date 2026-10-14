RCB host tree construction and the original tiled Step10 HIP force kernel
are preserved. HECBENCH_IRIS selects the original GPU host branches; DMem
buffers and synchronous IRIS tasks replace native pinned allocations, copies,
streams and kernel launches. The force kernel keeps original2D grids,
TILEX/Y4, BLOCKX32/BLOCKY4, float particle data and 64-bit identifiers.
Original eight/nine-argument driver semantics and make run defaults remain.
The MPI-serial native configuration is retained. CPU tree construction is
part of the original algorithm; validation uses the IRIS HIP RCB kernel.
HALO_DUMP_DIR exports all particle positions/forces per trial; CPU direct-force
reference and artifact comparison run after releasing the shared GPU lock.
