IRIS DMem epistasis port. Preserves seed 100, all three arguments, default
`146 31339 100`, bitpacking/popcount kernel, and the native minimum-score
index plus 1e-3 score reference check. `iris_rw` preserves FLT_MAX scores in
the unwritten lower triangle. Both inputs use `iris_r`; scores are flushed.

Build with `make`; `IRIS_ARCHS=HIP make run RUN_ARGS="146 128 3"` is a bounded
validation case. Evidence includes native HIP and IRIS PASS for 128 and 513
SNPs. The much larger default is preserved but was not run during validation.
Optional CUDA backend source is provided without a CUDA validation claim.
