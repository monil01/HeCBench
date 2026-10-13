All 20 original CRS coding kernels and the original host Galois/Jerasure
bitmatrix helpers are retained. Kernel symbols replace host launch-function
dispatch. Each input/parity task slice and the bitmatrix use DMem; parity
results are flushed before use. int parameters, unsigned bitmatrix words and
64-bit long data words retain their original widths.

Original dynamic shared memory is expressed as a static MAX_THREAD_NUM array
because the supplied IRIS task API does not take dynamic shared-memory bytes.
This exactly matches the original fixed 128-thread launch and workspace size;
coding operations, indexing and barriers are unchanged.

Default arguments remain 10 MB per block and 10 tasks. generateRandomValue
still calls srand(123) for each configuration, after IRIS initialization.
CRS_VERIFY=1 adds input and encoded-byte digests for every visited k/m/w case.
The isolated native HIP adapter adds only those same diagnostics and uses the
unchanged original kernels/helper sources; original source files are untouched.
Validation can use explicit bounded CLI values without enabling DUMP, so it
retains the entire default configuration sweep rather than a single variant.
