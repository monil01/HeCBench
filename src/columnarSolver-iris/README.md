# ColumnarSolver IRIS
Retains encrypted/expected text, bigram filename argument, 32x32 threads,
150000 climbing iterations, LCG setup and original PASS/FAIL plaintext check.
Three-character bigram input buffer includes required trailing NUL (original
buffer of two overflowed). DMem score/encrypted inputs read-only, RNG setup
writes state, decode reads state and writes candidates, then flushes candidates.
COLUMNAR_VALIDATION_CLIMBINGS optionally limits iterations; unset retains150000.
COLUMNAR_VALIDATION_DUMP optionally writes every candidate int for exact HIP
comparison. Original scoring archive recovered in local state, not source tree.
