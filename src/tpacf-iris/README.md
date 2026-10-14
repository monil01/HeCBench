TPACF IRIS DMem port
===================

Build with `make`; run with `IRIS_ARCHS=HIP make run`. IRIS_ROOT, IRIS_BUILD,
CXX, HIPCC, NVCC and RUN_ARGS are configurable. The original Makefile defaults
remain97178 requested data/random points,100 random catalogs,-b5 -l1 -u10000
-a -j10, and output small_results_test.dat. Input catalogs must be supplied;
the native parser's rounding of point counts down to a multiple of4 is retained.

All original stages are represented: symmetric ACF, cross/tiled ACF, packed
four-bin output, per-block GPU histogram, GPU histogram merge, host partial
histogram accumulation, padding correction, jackknife compilation and correlation
output. Pointer-bearing cartesian kernel arguments are flattened to separate
DMem arrays. The31 boundaries actually read by the original kernels are a DMem
input instead of a device constant symbol. Native kernels use128-point blocks;
their dynamic shared array is represented by an equivalent fixed128-element
shared array. Arithmetic, indexing, sentinel bins and native host formulas are
preserved. Native CUDA/HIP siblings are untouched.

HECBENCH_TPACF_HISTOGRAM_DUMP optionally records all64-bin DD/DR/RR partial
histograms in a binary artifact for verification. The native validation fixture
adds only the same output dump to an isolated copy of the original host main;
it uses unchanged native HIP kernels. This does not repair native algorithms.

Validation uses authentic catalogs recovered from gtcasl/gpuocelot at immutable
commit fa63920ee7c5f9a86e264cd8acd4264657cbd190, each with97178 rows and verified
Git blob/size/SHA256 provenance. CLI cases512 with one random catalog,1025 with
two catalogs, and97178 with two catalogs passed. Effective counts are512,1024,
and97176 respectively. Full requested-size validation exercises all six16384-
point tiles, including off-diagonal tile pairs and ten jackknife partitions.
All2048 raw histogram values match native HIP exactly; the entire8225-byte
correlation artifact is byte-identical. Verification also checks independent
DD/RR pair conservation, DR pair counts and every printed count/correlation.
CPU verification runs outside the shared GPU lock. No CPU fallback or vendor
GPU replacement is used. The default100-random-catalog workload is unverified;
only two authentic random catalogs are available and were used.

The histogram DMem flush currently transfers the whole allocated partial-
histogram buffer, rather than the native path's first32 histograms. This is a
transfer/performance limitation; host accumulation uses only the original
number of partial histograms. Native/IRIS timing parity is not claimed.
