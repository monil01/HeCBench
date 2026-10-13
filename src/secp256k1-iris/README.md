The original secp256k1 device arithmetic, 512-entry constant input table,
one-block/one-thread launch, 100-repeat default and exact 32-byte result check
are preserved. uint is explicitly 32 bits, ulong is 64 bits, and the original
storage structure layout is asserted to be 64 bytes.

IRIS DMem owns the input/output device storage, and the last task flushes the
output before the unchanged expected-result PASS/FAIL comparison. No RNG,
native allocation, native launch or replacement crypto implementation is used.
Validation builds the original native HIP source in an isolated copy, without
semantic adapters, and runs the original default repeat count.
