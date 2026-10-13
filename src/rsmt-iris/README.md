# rsmt IRIS / HIP
This port retains the CUDA routing parser, the one-file-name command line,
256-pin truncation, integer coordinates and short edge IDs, median construction
for three pins, Prim MST tie breakers, and iterative Steiner insertion.
There is one two-stage solve per invocation; the original has no RNG or repeat option.

Build with `make`, then run `IRIS_ARCHS=HIP make run RUN_ARGS=/path/input.gr`.
The default input remains
`../rsmt-cuda/newblue7.kraftwerk70.3d.80.20.82.m8.gr`.
Only its DVC pointer is present in this checkout, so default execution requires
fetching the dataset. No substitute is silently selected.

All accelerator arrays, the large-net worklist, and the three mutable scheduling
counters are IRIS DMem. Both kernels submit synchronously to the same HIP device.
Every result array is flushed to the host. Host storage uses ordinary allocation.
No native HIP allocation, copy, or launch API is used by the host.

P1 backend changes: CUDA's 32-lane intrinsics map to the native HIP sibling's
64-lane shuffle, ballot and population count, preserving that baseline's
wave-synchronous algorithm on gfx908. Kernel specializations use 12 waves x64
for the first stage and 3 waves x64 for the second. Launches retain MI100's
240-block schedule (2 x120 compute units). A device kernel probes wave size;
other wave sizes fail explicitly. HIPFLAGS can select another architecture, but
wave32 backends and NVIDIA are unvalidated and require a corresponding kernel
specialization. The occupancy warning for the large-net kernel also occurs in
the native baseline; no performance claim is made.

Optional `RSMT_DUMP=/path/artifact` observes every input/output offset, output
coordinate, and edge without changing the command line or normal totals.
Validation compares this artifact exactly with an isolated, dump-only native
HIP copy, also checking the untouched baseline's totals. Deterministic routing
fixtures cover duplicates, coordinate clamping, 2/3-pin cases, noncollinear
4..256-pin nets, the >32-pin worklist and 257-pin truncation. Independent checks
validate connectivity, tree size, input preservation, and Steiner improvement
relative to a host Prim MST. The missing 194 MB production input is not validated.
