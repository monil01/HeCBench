IRIS DMem port of the original complex arithmetic benchmark.

Both custom kernels and the original device-only complex reference kernels
run in single and double precision. Each output byte must equal five, encoding
the original five complex identities at tolerance1e-3. Reference kernels retain
the native inline HIP complex helpers (and CUDA cuComplex helpers for that
backend); no opaque host library operation is replaced.

RUN_ARGS retains10000000 elements and1000 repeats. Validation uses10000000
elements with one repeat on native HIP and IRIS_ARCHS=HIP. Every checksum byte
is checked for each of the four kernels; both warmup kernels are retained.
