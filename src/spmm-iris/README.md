IRIS DMem sparse-times-sparse multiplication with complete CSR output.

The explicitly authorized equivalent HIP implementation accumulates sparse
products in row-local dense scratch, then computes prefix sums and emits sorted
CSR offsets, columns and values entirely on the GPU. All device allocations and
dispatch use IRIS DMem/tasks. CPU multiplication is an independent oracle only.
P1: O(M*N) additional scratch and different vendor scheduling, summation order
and throughput. This is not a vendor-performance replacement.

Original seven arguments, srand(123)/drand48 initialization, CSR outputs and
verification tolerance 1e-2 are retained. Inputs are generated before IRIS init.
Both original default run cases remain; bounded tests do not validate defaults.
HECBENCH_DUMP_PREFIX exports complete CSR output for native HIP comparisons.
Native sources remain unchanged; observational native adapter only adds dumps.
The original NVIDIA notice is preserved in NOTICE and applies to adapted helpers.
