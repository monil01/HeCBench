The two original CUDA shared-memory algorithms are retained in
`original_kernels.hpp`. IRIS launches HIP kernels with DMem buffers, 512 threads,
8192 work items per block, original seed 2, warmup resets and timing accumulation.
Default runs remain 10000000 items, output size 32, dimensions 8/64, 1000 repeats.

The local IRIS HIP task interface cannot request ordinary launch-time dynamic
shared memory. Each algorithm therefore has static capacities of 1024, 4096 or
16384 float words; the host chooses the smallest capacity that contains the
original output and index buffers. Requests beyond 16384 words are rejected.
This accommodates the MI100 64 KiB shared-memory limit and preserves both default
cases. Rounding capacity can change occupancy, so timing includes this IRIS
portability compromise and synchronous task overhead. CUDA kernels are provided
but were not built or executed here.

Set `HECBENCH_VALIDATE_DUMP` to a path prefix to capture all output floats after
warmup, after kernel 1, and after kernel 2. Capture flushes are outside the timed
regions. Native comparisons use temporary copies; the native sibling is unchanged.
