IRIS DMem LU decomposition. Original -s/-i/-v flags, matrix construction,
block size16, kernel ordering and default make run -s8192 are preserved.
Kernel matrix_dim remains size_t (host launch passes an explicit size_t scalar).
The same iris_rw DMem matrix carries diagonal/perimeter/internal results across
tasks, with one final flush. -v retains the original reconstruction threshold
1e-4 and also reports PASS/FAIL; nonfinite reconstruction fails.
HECBENCH_DUMP optionally saves lud_result.bin for native matrix comparison.
Bounded validation uses -s32 -v and -s64 -v, with a test-only native HIP dump
adapter that does not change its kernels, matrix generation or verification.
