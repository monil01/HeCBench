# Popcount: exact UInt64 bit population, six algorithms

The original CUDA source remains unchanged and is included by the adapter.
The original Julia source and Makefile are preserved in `reference_original/`.
The named study uses 4,099 words, 256 threads per block and eleven internal
repetitions; the final block is partial. The input covers zero, all-one and
high-bit words, every one-hot and its complement, all 256 byte values at each
of eight byte positions, and deterministic full-width LCG words.

Every trial records actual device-input bytes and all six complete Int32
output arrays with two exact guard words per algorithm. The native oracle
clears the lowest set bit repeatedly. Julia independently sums all 64 bit
positions. These integer contracts require exact agreement. No floating-point
tolerance or checksum replaces full-output checks.

The measured region is a six-algorithm dispatch suite divided by internal
repeat count, with host-to-device LUT upload before every fifth-algorithm
launch and device-wide completion. Inputs, allocations, output resets,
readback, CPU oracles and artifact writes are outside timing. All five
trials are checked; only the first timing is excluded from the mean.

Native pc5 uses constant-memory LUT storage. The study Julia models use an
uploaded read-only global array; this storage/transfer implementation differs
and is an explicit performance covariate. The inherited CUDA.jl pc5 computed
byte counts using a population-count intrinsic instead of loading the native
LUT; its source is archived, and the study restores an actual uploaded LUT.
The exact bitcount operator is unchanged. Native has six device entry points;
the Julia implementations use one authored dispatcher kernel with six runtime
algorithm branches. The evidence does not certify every CLI length, the
original platform-specific `rand()` stream or general application behavior.
