IRIS DMem separable convolution. Seed 2009, radius 8, warmup, row/column tiled
geometry, original host reference and relative L2 <1e-6 verdict are preserved.
Default RUN_ARGS remains 8192 8192 1000. Valid images require width divisible
by 128 and height divisible by 64, as in the source row/column assertions.
Buffer is written by rows and read by columns on-device; final output is flushed.
Bounded validation uses 128x64/repeat1 and 256x128/repeat3 against native HIP.
Inherited NVIDIA copyright/EULA notices retained in all source derivatives.
