# geodesic IRIS port status

Status: verified against native HIP on the full original input.

All 12582912 output distances were finite and matched native HIP exactly.
Both models retain a maximum single-precision CPU-reference error of 30.25
meters. Numeric verification compares every GPU distance and the complete
nonfinite mask, without changing the native algorithm.

The original locations archive is identified by the DVC metadata in
`src/geodesic-sycl/locations.tar.gz.dvc`. Supply its extracted locations.txt
as the first argument; the second argument is the repetition count.

IRIS dmem preserves the full 2097152-city, six-reference calculation and
the 1000-repeat run default. Original host distance calculation remains the
independent reference; bounded validation uses one repetition with every pair.
