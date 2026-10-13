# ColumnarSolver IRIS
Retains encrypted/expected text, bigram filename argument, 32x32 threads,
150000 climbing iterations, LCG setup and original PASS/FAIL plaintext check.
Three-character bigram input buffer includes required trailing NUL (original
buffer of two overflowed). DMem score/encrypted inputs read-only, RNG setup
writes state, decode reads state and writes candidates, then flushes candidates.
COLUMNAR_VALIDATION_CLIMBINGS optionally limits iterations; unset retains150000.
COLUMNAR_VALIDATION_DUMP optionally writes every candidate int for exact HIP
comparison. Original scoring archive recovered in local state, not source tree.

Validation status: BLOCKED, not a verified working port. At4096 climbs every
candidate matched native HIP but both plaintext checks printedFAIL. Native
150000/50000 searches timed out at60/45 seconds; corresponding IRIS runs failed
task submission. Defaults and RNG unchanged. Full evidence and two attempt
snapshots are in .porting-state/porting_logs/columnarSolver-iris/manifest.json.
