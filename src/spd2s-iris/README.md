IRIS DMem dense-to-CSR conversion replaces vendor conversion with HIP row counts,
a GPU integer prefix scan and row-local scatter. Counts/offsets/columns retain
int64_t widths; dense/values float; columns emitted row-sorted. Original seeded
initialization (srand123,srand48123), arguments M N nnz repeat and both default
Makefileinvocations preserved. Original CPU reference and sorted global CUDA
verification retained; native HIP additionally checks per-row ordering exactly.
Count/prefix analysis precedes conversion timing as in source library workflow.
Single-thread GPU prefix is a portability/performance compromise, not CPU work.
All buffers DMem/iris_rw and explicit output flushes. CUDA backend unvalidated.
