Bitcracker uses IRIS data memory for the SHA256 schedule, password batches,
encrypted inputs, and the result. Both original device kernels are retained,
including the million SHA256 rounds, launch geometry, integer widths, and
the original non-atomic `*found += 1` behavior. The batch default is 60000.
Host changes to password batches are communicated through DMem.update().

`BITCRACKER_VERIFY=1` adds a digest of all 67108864 schedule words and the raw
found counter. Normal output and arguments are preserved. Validation uses
two original dictionary entries with `-b 1` and `-b 2`; these are explicit
bounded runs, not default replacements. A local native HIP adapter adds only
these diagnostics and leaves the original HIP files untouched. No RNG is used.
