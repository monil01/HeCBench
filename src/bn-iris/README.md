# BN IRIS dmem port

See BASELINE_FIX.md for documented native/IRIS initialization, argmax and parent
decoding repairs. Full scores and graph artifacts must agree with native HIP;
verify_scores.py additionally uses independent contingency counting and lgamma.
Untouched native snapshots, failed translations and all diagnostics remain in
local porting state. No CPU execution fallback is used.

Preserves original Bayesian-network score enumeration, shared-memory max reduction,
CPU references, graph search100iterations, RNG seed2, float/int/bool widths, CLI
and default45nodes/600samples. All GPU allocations, transfers and launches are
IRIS dmem; partial writes rw, host updates preserve GPU data then call update.
computeKernel fixed256-float shared storage matches original256-thread launch.

make builds original defaults; make run uses original result10 CLI.
For bounded validation, make EXTRA_CFLAGS=-DHECBENCH_BN_VALIDATION uses the first
12columns and96rows of the unchanged checked-in original45x600 dataset.
Only bounded input dimensions change; scoring algorithm/search iterations
and default build behavior unchanged. Isolated native uses the same documented
input adapter. HECBENCH_OUTPUT optionally captures all local score floats;
original output file captures3 complete graph adjacency matrices and scores.

IRIS_ROOT defaults /tmp/iris; HIP only, prints backend3/type256 AMD proof.
kernel.cu retains CUDA-compatible kernels (CUDA not locally tested).
Original memset(D_Score,-999999.f,...) byte-fill semantics retained, not changed
to a floating-point fill. IRIS initialization precedes original srand(2).
