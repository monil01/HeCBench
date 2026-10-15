# KMC / LIBSVM with IRIS DMem HIP

The complete original dense LIBSVM parsing, precomputed RBF matrix and
cross-validation pipeline is retained. Every kernel-matrix column calls the
original real hipBLAS SGEMV through the shared IRIS vendor bridge, on
DMem-resolved device pointers and the IRIS stream. There are no native GPU
allocations/copies in the port. Existing CPU norm/exponential/SVM operations
remain on the CPU exactly as in the CUDA/HIP source; they are not substitutes
for the offloaded matrix-vector products.

All options, default gamma, labels, row-identifier convention, and default
make-run `-v 10 gisette_scale` remain. IRIS initialization can touch libc RNG,
so srand(1) restores the original default C RNG stream before the unchanged
LIBSVM driver runs. Optional KMC_DUMP=1 prints every RBF matrix element and
every cross-validation prediction. KMC_VALIDATE=1 independently verifies the
entire RBF matrix using long-double dot products of the actual float inputs
(atol1e-5/rtol1e-4); the GPU result is never replaced with that reference.

P1 timing: the callback creates/destroys a hipBLAS handle per column and each
synchronous DMem task flushes its result for the original host exponential
calculation. Original kernel-only performance is not claimed. Ported CPU
LIBSVM files differ only in whitespace and optional prediction diagnostics.

Validation uses unchanged original GISSETTE rows recovered from the official
LIBSVM dataset host, not the repository's misleading 84-byte named fixture.
Local provenance and hashes document the partial upstream bzip2 download and
the exact complete-row crops. The full default dataset and default-size
cross-validation performance are not claimed validated. No datasets or
generated output artifacts are committed.
