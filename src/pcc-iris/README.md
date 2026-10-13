# PCC IRIS
Preserves N/L arguments, seed123 default_random_engine with uniform[-6,6], CPU
row preprocessing, single/multi-pass4GiB threshold, packed upper triangle and
small-case corrs.txt output. BLAS product replaced by HIP gram kernel through
IRIS tasks (P1 performance/reduction-order difference). DMem holds normalized
input, chunk matrix and output; original extraction kernels retained. Optional
PCC_VALIDATION_MEMORY_BYTES forces the original budget decision at boundedN.
Default run90112 165 retained; only bounded numeric arrays are validated.
