IRIS DMem SU3 matrix multiply. Original flags, lattice initialization, warmups,
default make-run arguments, grid (total_sites blocks) and all 1e-6 real/imaginary
assertions are preserved. Native -d is parsed but unused; the IRIS port likewise
selects the first actual HIP GPU. The MILC_COMPLEX representation is the default
baseline configuration; PRECISION=1 by default, PRECISION=2 is supported by the
build. Host/device static assertions require matching site size (320/640 bytes).
Inputs use iris_r, output iris_w, and a final task flushes the full lattice.
PASS is emitted only after the original assertions succeed (do not use NDEBUG
for validation). Bounded runs should use -l 4 -i 2 or -l 8 -i 3.
