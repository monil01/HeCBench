IRIS DMem wlcpow port. Seed 19937, n=1000000, n_type=32, input generation,
period, default repeat 1000, checksum outputs and force accumulation are preserved.
Forces start with host zeroes transferred through iris_rw DMem and remain on the
GPU across iterations; only final outputs are flushed. Static shared storage
33*8 floats matches the fixed n_type=32 native allocation. Scalar float3 and
array float4/int2 use the matching HIP ABI on the host. Nonfinite forces fail.
Bounded validation uses repeats 1 and 2 at the original fixed problem size,
comparing all three native checksum values (rtol=1e-5, atol=1e-5).
