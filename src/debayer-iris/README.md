Debayer IRIS DMem port. Preserves RGGB, seed 123, 32x5 tile geometry,
argument order and default 8192 8192 100. Numeric baseline: original checksum.
All buffers are conservatively iris_rw; copies update DMem or flush to host.
HIP tasks submit synchronously; printed timings include IRIS task overhead.
Run with IRIS_ARCHS=HIP and hold the shared validation flock. CUDA unvalidated.
