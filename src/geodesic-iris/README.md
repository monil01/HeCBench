# geodesic IRIS port status

Status: deferred.

The native HIP baseline builds, but its run target requires
`src/geodesic-sycl/locations.txt`. That dataset is not present in this
workspace; `src/geodesic-hip/README` says to extract `locations.tar.gz` in
`geodesic-sycl` before running the benchmark.

Because native HIP cannot run locally, this IRIS port is deferred rather than
validated or partially translated.
