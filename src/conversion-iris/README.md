# conversion IRIS port

This port uses IRIS data memory (`iris::DMem`) for all source and destination
buffers and launches HIP backend kernels with concrete symbols for each
conversion pair.

The native CUDA/HIP benchmark is a bandwidth microbenchmark and does not
initialize source buffers or check converted values. The IRIS host initializes
deterministic small integer-valued inputs and prints a `PASS`/`FAIL` line after
each conversion so the port has value-level correctness evidence in addition to
the original bandwidth line.
