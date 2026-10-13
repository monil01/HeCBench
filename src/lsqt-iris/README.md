LSQT keeps all 14 original device kernels and the original model,
Chebyshev/time-evolution, DOS/VAC/MSD/spin/LDOS host algorithms.
All device ownership is DMem, host uploads call update, partial and in-place
writes use rw, and copy_to_host flushes before copying to the original host
consumer. Vector swaps exchange DMem ownership, not numerical data.
Default double precision, native -ffast-math and DEBUG seed12345678 retained;
USE_SP remains an explicit source/build option. IRIS srand(time) does not
affect the original std::mt19937 generator reset by Model.

P1: CUDA warp synchronization maps to the HIP wave-barrier builtin; the
original reduction arithmetic/indexing is unchanged. Native original kernels
are retained unmodified in an isolated native build for comparison.

Original default driver references recovered DVC example directories. Inputs
are not vendored or overwritten; RUN_ARGS is configurable for an isolated
path-only driver adapter. Full original4-million-atom defaults are attempted
with a bounded timeout; separately labeled bounded lattices use128x128 and
16384x1 with all other original input parameters/times/energies retained.
Original archive MD5c7d1af7bc30a4faf929c8b399c071024,35364096bytes, safely
verified/extracted only in owned local state. Prior missing-data history stays.
Numerical artifact tolerance: atol1e-10 + rtol1e-8, every emitted finite value.
