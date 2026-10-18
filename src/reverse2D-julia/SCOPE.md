# Exact matrix-axis reversal, original scalar and corrected vector strata

The historical CUDA.jl source only ran a smoke marker and printed placeholder
text. Its exact source and Makefile remain in reference_original/. New
CUDA.jl, KA and JACC programs perform the actual coordinate permutation.

Two named contracts each cover 32 cases: UInt8/16/32/64, row/column storage,
row/column reversal, and two shapes. All actual device-input bytes and all
output bytes plus 64 guard bytes per allocation are captured, compared
exactly and checked by an independent Python coordinate oracle. Full-width
integer fixtures expose byte/lane errors hidden by the original 0..255 range.

The primary scalar contract uses odd17x19 and33x35 shapes. Unchanged native
headers select vector length1, and every model passes all five full captures.
The corrected secondary contract uses32x64 and64x96, both safe for all
selected vector alignments. Actual unchanged native vector runs fail all
five checks, with244856 wrong output bytes in each capture. The original
coalesced-dimension loop swaps each vector lane pair twice. A minimal
Ratio/2 loop-bound correction restores intended element-wise reversal;
source originals, exact patch, rejected captures and an independent CPU
model of the faulty lane permutation are retained. Corrected vector results
must not enter untouched-native aggregates or erase the original failure.

The timed region is the same32-case kernel-launch suite, seven internal
repetitions and a device-wide completion; allocation/reset, copies, CPU
oracles and artifact writes are excluded. Original native wrapper chrono
and per-launch synchronization are replaced by this declared common batch
protocol. Five checks are retained, mean uses times2..5. Native paired
vectorized scatter and Julia scalar output-gather differ in work mapping
and access width; no causal abstraction ranking follows from these times.
Other shapes, in-place execution and the original full CLI are not certified.
