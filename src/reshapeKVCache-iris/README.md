# reshapeKVCache IRIS HIP
IRIS DMem manages all five buffers; the two HIP-specialized kernels preserve token-block geometry and original vectorized/scalar layout conversion. Output caches use iris_rw to preserve zero-filled padding; every warmup/timed launch is a separate synchronous task. Timing includes IRIS scheduling overhead.
Arguments, seed 42, shuffled slot mapping, 30 warmups, FP32/BF16 cases, block sizes 16/32, divisors 1/2/4/8 and default Makefile arguments 1024 32 128 1000 are retained. Complete byte comparisons to the host reference run after warmup and timed repeats; a complete-cache hash provides external HIP comparison.
The AMD implementation deliberately follows the existing HIP sibling's E4M3 FNUZ conversion (HIP software conversion on gfx908). CUDA uses NVIDIA E4M3, so raw encoding is backend-specific; no CUDA numeric validation is claimed.
Build: make. Run: IRIS_ARCHS=HIP make run RUN_ARGS="37 3 24 2". Requires an IRIS HIP GPU. No CPU fallback is accepted.
