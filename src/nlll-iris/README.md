# nlll IRIS port

IRIS `DMem` port of the negative log likelihood loss benchmark. The host uses
the same deterministic input generation and reference check as the CUDA/HIP
sources, and the HIP backend exposes one named kernel per data type and thread
block size.
