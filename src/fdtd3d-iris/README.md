Retains CUDA CLI/defaults, deterministic srand(0) input, host reference and
original shared-memory finite_difference kernel. Three buffers use dmem;
output uses rw to preserve halo and leading128-byte alignment padding.
Pointers alternate each timestep, with the actual last output flushed.
IRIS initializes after input generation/reference execution to preserve RNG.
HECBENCH_ARTIFACT optionally dumps every halo/interior float for full native
comparison. The isolated native copy adds only the dump. Validation uses
source-supported96^3 dimensions and3 timesteps; original tolerance1e-4
checks all interior cells and artifact comparison covers halo as well.
