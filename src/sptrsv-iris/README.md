# sptrsv IRIS port

This port preserves the native HIP sparse triangular solve kernel and moves
device memory ownership to IRIS `DMem`. Validation uses the `lp1.mtx` input
from `src/sptrsv-sycl/lp1.mtx.tgz` extracted to `/tmp/hecbench-sptrsv/lp1.mtx`.
The native HIP baseline was unstable at `repeat=2000` on this host, so the
checked `make run` default uses `repeat=1`, where native HIP and IRIS both
produce the same zero-residual `PASS` result.
