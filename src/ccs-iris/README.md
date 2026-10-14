CCS IRIS port
=============

Build with `make`, then run with `IRIS_ARCHS=HIP make run`. IRIS_ROOT and
IRIS_BUILD select the IRIS installation. RUN_ARGS accepts the original CCS
options; the run target retains the CUDA Makefile's workload and 100 repeats.
The original dataset must be available at the path given with `-i`.

The kernel retains the original one-thread block geometry, correlation,
augmentation, and sample-vector indexing. The host also retains the original
choice of temporary membership arrays for printing results. This port does
not repair those native algorithm choices. All device buffers use IRIS DMem;
the final task flushes scores, counts, and temporary membership arrays.

Validation uses the DVC dataset with checksum
838eacfcf0f945b7186e7ed7e25f1b1c and compares the BiBench-format artifact
against unchanged native HIP with the same options. Reduced repetition counts
bound validation time without changing the search coverage.
