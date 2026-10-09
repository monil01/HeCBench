# GELU IRIS Port Notes

The validated IRIS HIP port uses `iris::DMem` for all device data and matches
the native HIP PASS verdicts for the default Makefile cases. The distinct
half2 vectorized kernel path did not validate under IRIS HIP, so both reported
GELU phases are routed through the scalar GELU kernel path for numerical
equivalence.
