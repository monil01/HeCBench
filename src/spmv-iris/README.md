# IRIS DMem SpMV

Original nnz/rows/repeat CLI, srand48(1<<12), srand(123), float precision,
matrix/vector input helpers, size_t CSR/COO indices, dense GPU reference and both
CSR/COO error-rate checks retained. All device ownership/launches use DMem/tasks.
Original input generation occurs before IRIS initialization.

P1: authorized mathematical replacement of vendor SpMV uses one HIP thread per
row. COO locates sorted row runs by binary search. Different scheduling and speed
from vendor libraries; no vendor throughput parity. Dense reference kernel retains
the original source algorithm. Bounded whole-output native comparisons exercise
both formats, empty rows, non-block-aligned sizes and multiple repeats. Default
16777216 nnz / 10240 rows / 100 repeats retained but not validated.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS="350 67 2"`.
Optional HECBENCH_DUMP_PREFIX exports dense reference and complete CSR/COO outputs.
Original CUDA/HIP files are untouched; no CUDA backend is supplied.
