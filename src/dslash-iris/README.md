# Dslash with IRIS DMem

Retains original gather mappings, backward-adjoint and dslash kernels, CPU
reference assertions, single precision lattice 32^4 and 100 default iterations.
Kernel even-site scalar is explicitly int (native kernel parameter width), while
indices retain size_t. Partial backward/output writes use DMem read/write.
Optional HECBENCH_SEED seeds the original mt19937 for deterministic validation;
absent it retains random_device initialization. HECBENCH_DSLASH_OUTPUT dumps all
even-site complex vectors after reference checking. Native isolated adapter only
adds these seed/output/verdict hooks. Bounded validation uses LDIM=8,
ITERATIONS=3, PRECISION=1 and strict HIPFLAGS=-O3 in both builds.
