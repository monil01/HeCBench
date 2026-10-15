# ColumnarSolver IRIS
Retains encrypted/expected text, bigram filename argument, 32x32 threads,
150000 climbing iterations, LCG setup and original PASS/FAIL plaintext check.
Three-character bigram input buffer includes required trailing NUL (original
buffer of two overflowed). DMem score/encrypted inputs read-only, RNG setup
writes state, decode reads state and writes candidates, then flushes candidates.
COLUMNAR_VALIDATION_CLIMBINGS optionally limits iterations; unset retains150000.
COLUMNAR_VALIDATION_DUMP optionally writes every candidate int for exact HIP
comparison. Original scoring archive recovered in local state, not source tree.

Validation status: verified on MI100 using HIP-only IRIS DMem. Both native HIP
and IRIS complete the original150000-climb search and printPASS (54.28/58.40s
validation wall times respectively). All406528 candidate integers match exactly;
every candidate preserves the ciphertext letter histogram. Independent CPU
sequential-float32 scoring of all candidates chooses index361, score3066.79321,
and all397 characters exactly match the original expected plaintext.

Documented minimal CUDA/HIP/IRIS source repairs: bound bigram tokens to2 letters
with3-byte storage; clamp only LCG float conversions that round to1.0 to the
largest float below1 (0x1.fffffep-1f). Unsigned LCG state updates, initial seeds,
search operations and default150000 iterations remain unchanged. The original
endpoint can index key[30]; previous undefined behavior is not a valid numeric
reference. Its causal connection to earlier timeouts was not established.
Untouched sources and all historical failed attempts remain in local state.

Original scoring archive MD5 bbcba6be85e7d0d1cf26a58d76fea8d0 matches the DVC
pointer (6100bytes); unchanged recovered data is restored locally to the
original default input path but is not committed. BENCHMARK inputs remain the
original ciphertext, plaintext and676 scores, not fabricated replacement data.
CPU candidate/score/text checks execute after releasing the bounded native/IRIS
pair GPU lock. No CPU search fallback. NVIDIA execution was unavailable.
