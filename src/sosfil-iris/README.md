IRIS DMem SOS filtering retains float/double CPU checks (atol=rtol=1e-4),
30 warmup iterations, repeat argument and fixed default 8 signals x 1000000 samples.
Original DEBUG configuration is 2 signals x 65 samples. Compile both host/kernel
with -DDEBUG for bounded validation. Shared memory is a static array per kernel
specialization, sufficient for the unchanged 64 sections, 2 zi entries and 6 coefficients.
DMem host refresh uses update(host), intermediates use iris_rw, tasks synchronize.
CUDA kernel supplied unvalidated. Timing includes IRIS overhead.
