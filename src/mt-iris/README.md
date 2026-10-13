# Mersenne Twister IRIS dmem HIP port

Preserves both native GPU kernels, source CPU reference, 4096 streams, seed777, 5860 values per stream, fixed128-thread workgroups, Gaussian Box-Muller transform, mandatory repeat argument and original L1<1e-6 check. All device memory and dispatch use IRIS DMem/tasks. The run target retains1000 repeats.

The original data/MersenneTwister.dat and data/MersenneTwister.raw files are required in the working directory. They were recovered from the repository DVC remote using exact archive MD5 2d20b003f0064b7fd8fed7cb394a955e (79247 bytes), not synthesized. Validation assets/binaries stay outside committed source. A runtime directory needs kernel.hip plus data/; main may be invoked by absolute path.

Build make; run IRIS_ARCHS=HIP ./main <repeat>. Configure IRIS_ROOT/IRIS_BUILD/CXX/HIPCC. Optional HECBENCH_DUMP_PREFIX exports the complete float32 GPU result without changing computation. Original CUDA/HIP siblings are untouched; original source notices are retained.

No NVIDIA backend or full1000-repeat production performance claim. Synchronous task timing includes IRIS overhead; default problem size is retained. Invalid nonpositive repeats are rejected rather than retaining undefined/uninitialized source behavior.
