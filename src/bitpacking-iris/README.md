IRIS DMem host and HIP kernels for the int32_t bitpacking benchmark. Defaults
retain srand(0), offset 87231, 13-bit inputs, all six original sizes, and 1000
iterations. Run bounded validation with `make run RUN_ARGS="--max-size 83621"`.
The bound selects a prefix of the original size list without changing its data.

The three GPU stages compute min/max, derive width, and pack the same 32-bit
little-endian words as CUDA/HIP. Packing uses independent output-word threads
instead of shared input staging. All output storage is flushed through DMem.
The original sampled unpack check is retained; sizes <= 872163 additionally
check every element. CHECK lines expose minimum, width, and FNV-1a packed-byte
hash for comparison with the unmodified HIP compression implementation.
Kernel timing includes synchronous IRIS task overhead; native HIP timing does
not. CUDA kernel compilation is optional and has not been validated locally.
