IRIS DMem Needleman-Wunsch port preserving dimension/penalty/repeat arguments,
seed7, BLOSUM62,16-thread wavefront tiles,100warmups, default16384/10/100, and
full integer-matrix memcmp verification against the original host reference.

Both wavefront kernels use typed dispatch and synchronous IRIS HIP tasks.
The partially written matrix remains rw across stages and iterations, while
the scoring reference is read-only. Final matrix flushes before verification.
Validated32x32 and128x128 with penalty10/repeat2; both HIP and IRIS full checks
PASS. Large default size is preserved but not GPU-validated in this pass.
