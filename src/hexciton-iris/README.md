All 25 original CUDA kernel variants are retained as IRIS HIP tasks. Input
layouts, compile-time settings, reference calculation, deviation summaries,
and the default 1001 iterations are preserved. Output resets use DMem.update
and read/write access because several variants accumulate or partially write.
The final iteration flushes the result for the original numerical comparison.

CUDA's default vector width is 4; the native HIP sibling changed it to 8
while retaining four-lane vector types. Validation uses an isolated native
HIP copy with only that macro restored to CUDA's default 4. Original HIP
files remain untouched. No seeds or input dimensions are changed. Bounded
validation may override the existing NUM_ITERATIONS macro explicitly; the
production Makefile leaves its original default in effect.
