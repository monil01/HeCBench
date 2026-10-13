IRIS DMem in-place512 FFT/iFFT retains double default (tolerance1e-6), optional
SINGLE_PRECISION (1e-4; define for host and HIP kernel), selector0..3 sizes1,8,96,256MiB,
pass argument, seed2, thread geometry64 and CPU forward/inverse checks.
Portable host complex pairs have matching8/16-byte alignments to HIP vector ABI.
In-place buffer iris_rw retained through forward, flush, inverse and timed iterations;
host inputs update DMem explicitly. Default run3 100 preserved. CUDA unvalidated.
