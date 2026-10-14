The host owns all buffers through IRIS DMem. Contiguous tensor accessors are
represented by raw kernel pointers and the original dimensions; padding,
stride, dilation (one), bias, RNG seed, and multiplier/kernel sweeps are retained.
The default workload is unchanged. GPU results also receive a complete scalar
reference check (absolute tolerance 1e-5).

Bounded validation: native HIP and IRIS `./main 2 3 17 19 2`; compare all
12 native checksums with absolute/relative tolerance 1e-5. IRIS_ARCHS=HIP.
