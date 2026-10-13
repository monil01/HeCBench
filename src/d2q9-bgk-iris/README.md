# d2q9-bgk, IRIS DMem / HIP

The original fused lattice-Boltzmann kernel and all CPU initialization,
reductions, Reynolds calculation and output formatting are retained. Device
SoA buffers are DMem double buffers; partial-sum arrays use read/write access
because each timestep writes only its own slice. All outputs are flushed before
CPU use, including the correct swapped storage on odd iteration counts.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS='input.params obstacles.dat'`.
Original default is the 256x256 input, supplied via its original DVC archive.
No input or reference data is committed. Original warmup count is 1000;
validation uses parameter files with more than 1000 iterations. Original grid
assumptions remain: nx is divisible by 128, ny is positive. The source's
transposed obstacle flag indexing in its output format is also preserved.

`D2Q9_DUMP=/path/file` optionally writes all final nine float populations in
cell-major order. Native comparison uses an isolated HIP copy with the same
lossless observer; native sibling files are untouched. Original output files
final_state.dat and av_vels.dat remain unchanged. Validation compares all raw
populations, every output row and every average-velocity iteration; it also
runs recovered reference checks where supplied by the original dataset.

`D2Q9_GPU_LEASE=1` serializes only the GPU routine, never CPU preparation,
reduction, output or external verification. Queue wait is capped at 300 seconds
and each GPU phase at 90 seconds, with measured wait and held times.
