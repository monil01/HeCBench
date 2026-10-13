Preserves all four CUDA SPH kernels, geometry construction, struct fields,
particle counts, Euler seed, timestep and output CSV. Three buffers use dmem;
particle updates use rw and the final update flushes the full fluid state.
Default steps=500 remains. VALIDATION_STEPS=3 bounds native/IRIS checks without
changing kernel arithmetic or initial data. HECBENCH_ARTIFACT dumps all14
double fields per particle (with an int32 particle-count header); the isolated
native copy adds only the same dump and bounded-step option. Full finite
particle fields and CSV output are compared, including positions/velocities.
