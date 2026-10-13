IRIS dmem port preserving original integration, spatial hashing, bitonic sorting,
cell reordering and collisions. Default 16384 particles, 64^3 grid, timestep,
seed1973 and 10000 run iterations are unchanged. IRIS initializes before
initGrid reseeds RNG1973. Host vector types preserve the native struct ABI;
device compilation asserts simParams_t is104 bytes. Partial writes use rw.

PARTICLES_DUMP optionally writes all final position/velocity components, sorted
hashes and indices. Isolated native validation adds only the identical hook,
leaving native siblings unchanged. Bounded tests use 1 and3 CLI iterations
at original particle/grid sizes and compare complete artifacts. Shared hash
storage is an equivalent fixed65 elements for the original64-thread kernel.
CUDA kernels retained but not validated on NVIDIA hardware.

