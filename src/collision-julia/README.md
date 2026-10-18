# collision: source warp duplicate detection

The named study covers40 full32-lane Int32 inputs:32 source duplicate-count fixtures frozen from the current host libc srand123/rand stream, plus eight signed/extreme/nonadjacent cases. Both native kernels and device globals remain unchanged. All complete flags, masks and actual guarded input echoes are checked bitwise against both references and independent CPU oracles. Native outputs have no output guards; only actual input guards are claimed.

Historical CUDA.jl has a genuine quadratic/serial implementation of the same operator. Its original source/Makefile are preserved byte-identically; the new study reference restores native warp-bitonic mapping. KA/JACC use their real indexing/dispatch with explicit CUDA shuffle/vote backend extensions, sharing one audited network helper. This is intentionally a CUDA-specific hybrid, not proof of a universally unsupported portable feature.

Five outer trials each sum40 host-inclusive100-repeat two-kernel batches divided100; only trial1 timing is excluded. Warmup, allocation, transfers, output reset and capture are outside each region. Per-case times are in path_timing_summary.json. This differs explicitly from source CLI timing, which includes generation/allocation/transfers.

4522 UInt32 words per trial retain input bits, all1280 flags/40 masks and1920 device input echoes including640 guards. The serialization mask marks every occurrence after the first original lane for each key; it is not a mask of all collision participants. scientific_witness.json records the source-comment counterexample and support boundary. Run audit_saved_outputs.py on a saved bundle to independently recheck full arrays, masks, device echoes and path sums without a GPU.

First candidates/checks passed for all current versions; no failure or repair is fabricated. Token boundaries record shared helper/reference preparation separately from ordered framework wrapper work. Root owns commits.
