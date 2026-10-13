# Lanczos, IRIS DMem / HIP

Original graph/CSR construction, basis initialization, recurrence, dot-product
reduction order, step sweep, CPU Lanczos and TQLRAT/spurious removal are retained.
The original four template kernel bodies are device helpers behind eight C
exports (float/double). There are no opaque BLAS/sparse-library replacements.
All GPU arrays are DMem; updated vectors are read/write, full products write-only.
Partial dot sums are flushed and reduced in the original host order. Three
DMem vector roles swap without resetting their contents.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS='-g graph.txt -n 800000 -k 3 -d float'`.
The default command is identical to the native Makefile: `-d` is a flag and
actually selects **double**, regardless of the ignored positional `float`.
Omitting `-d` selects float. CLI defaults/options are unchanged. The graph reader
ignores edges with vertex numbers beyond `-n`, permitting bounded original-graph
tests without generated graph replacements. Data is not committed.

`LANCZOS_DUMP=/path/prefix` collects all alpha/beta recurrence coefficients
(including the last beta before resize) and all GPU eigenvalue lists.
`LANCZOS_DUMP_VECTORS=1` additionally captures every final x, previous-x and y
element for each step sweep. Lossless hexadecimal artifact I/O occurs only
after the GPU lease, before the unchanged full CPU reference sweep.
Native HIP comparison uses isolated copies with these observers only.

`LANCZOS_GPU_LEASE=1` takes the shared flock for the bounded GPU phase only,
including necessary scalar feedback and small tridiagonal postprocessing.
Queue wait is capped at 300 seconds, the entire GPU sweep at 90 seconds.
Graph loading, artifact I/O, full original CPU reference and verification are
outside the lease. Wait and held time are logged. IRIS init is after graph
loading; source basis initialization is deterministic and uses no srand.
