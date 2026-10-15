# d3q19-bgk IRIS DMem
The native CUDA/HIP boundary_values buffer is initialized to zero immediately
after allocation, matching its masked-OR accumulation's empty-bitset identity
and the IRIS DMem initialization. Previously undefined allocation garbage caused
native regression failure. Untouched native snapshots and old failed attempts
are preserved in local porting state. No physical equations, walls, defaults or
numerical tolerances are changed.
Original N argument/default128, ten runs of 2000steps with1000warmup, moving-lid configuration, equilibrium/stream/collision equations, population swaps and iteration149 energy regression for N102 are retained. No RNG is used.
Device pointer structs are reconstructed only inside kernels from explicit DMem parameters. Two read-only DMem tables replace CUDA constant-symbol uploads. Velocity is a contiguous three-component DMem. Population outputs use iris_rw to retain boundary populations; velocity uses iris_rw for selective output writes. One kernel per task, explicit velocity flush for host output, no CPU backend.
Optional validation environment HECBENCH_LBM_STEPS/RUNS/WARMUP bounds work without changing original defaults. HECBENCH_LBM_DUMP stores complete output velocity arrays; finite checks and full-precision energies accompany original regression.
Unused original host scratch arrays are omitted. Unused g buffer has zero host DMem backing but is never passed to kernels. Boundary values/directions are explicitly zero-backed: original native source leaves these uninitialized before masked OR updates; this removes undefined initial garbage without changing the intended physical conditions. Native regression and all output-array comparisons must pass before counting this port.
make; IRIS_ARCHS=HIP make run RUN_ARGS=102 (with validation environment when desired). HIP-only AMD validation; CUDA kernel provided but not tested. Timing includes IRIS overhead.

Validated after the native repair:15 full velocity fields /10,343,793 finite
values byte-exact versus HIP; independent sequential kinetic-energy reductions
exact and originalN102 regression passes1e-7. N8 retains full2000steps,1000warmup,
10runs; N17/65/102/128 use explicit160step validation controls. DefaultN128 full
2000step/10run runtime is not claimed.
