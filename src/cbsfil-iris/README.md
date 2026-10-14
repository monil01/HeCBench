IRIS DMem port of the original row/column cubic B-spline coefficient filter.

Build with `make`, then `IRIS_ARCHS=HIP make run`. RUN_ARGS retains the original
8192-by-8192 image and 100 timing repetitions. Validation uses the same full
image with one timing repetition, comparing native HIP and IRIS against the
original seeded CPU reference at absolute tolerance 1e-3. IRIS also rejects
nonfinite reference/device values. Row and column recursive passes, seed123,
input distribution, indexing and launch geometry are preserved.
