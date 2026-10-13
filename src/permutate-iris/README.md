# Permutation IID test, IRIS dmem / HIP

The original CPU statistics, bzip compression, shuffle RNG, device statistics,
10,000-permutation limit and early stopping rules are retained. Four original
HIP kernels have C-linkage exports. GPU storage and transfers use only DMem;
ranking counters are read/write and flushed after every batch. IRIS initializes
inside the GPU phase; the subsequent CPU compression RNG restores source seed
123, so runtime initialization cannot change its random sequence.

`make`, `IRIS_ARCHS=HIP make run RUN_ARGS=/path/to/input`.
The normal input contract is **one byte per symbol**, at least 1,000,000 bytes;
the existing CLI hard-codes 8-bit symbols, 1,000,000 samples, 10 blocks and 250
threads (2,500 parallel permutations). The filename `truerand_1bit.bin` does
not cause bit unpacking or select the separate 1-bit kernel path. Default input
is the native sibling's dataset path; no dataset is committed here.

`make -B VALIDATION=1` is an explicitly validation-only build: 4,096 samples,
4 blocks and 16 threads, with the original 10,000 limit and stopping logic.
`PERMUTATE_VALIDATION_BITS=1` enables the existing binary-symbol path only in
this build. Physical inputs must still be at least 1,000,000 bytes. Production
defaults and CLI are unchanged by the normal build.

Optional `PERMUTATE_DUMP=/path/prefix` records all 19 original statistics as
hexadecimal doubles, mean/median, all 57 final integer counters, IID verdict,
and every batch's complete 54 GPU counters. `PERMUTATE_SHUFFLE_DUMP=/path/file`
additionally records the complete final interleaved shuffled buffer and packed
binary buffer; use this only for bounded cases to avoid huge transfer files.
These observers neither replace nor alter the original computation.

For serialized local validation, `PERMUTATE_GPU_LEASE=1` acquires the shared
validation flock only inside the GPU routine: wait is bounded to 120 seconds
and each GPU phase to 90 seconds. CPU input/statistics, compression and external
verification run outside the lease. Wait and held-lease times are printed.
Normal runs do not automatically take this site-specific validation lock.

The compression message allocation reserves one extra byte for `sprintf`'s
trailing NUL (the source binary-symbol path otherwise overruns its allocation).
The compressed payload, length, bzip algorithm and statistical outputs are
unchanged. The isolated native validation copy has the same memory-safety fix;
the native sibling itself is untouched.

Validation compares the original HIP kernels in an isolated, equivalently
instrumented native snapshot against IRIS, including complete artifacts and
native IID verdict. The original DVC asset was recovered and verified in local
scratch. Bounded fixtures are supplemental kernel checks, not a production
input or general NIST/entropy-quality certification.
