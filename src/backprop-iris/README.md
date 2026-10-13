IRIS DMem port of CUDA backprop, validated against the native HIP sibling.
Build with make; run with IRIS_ARCHS=HIP make run. RUN_ARGS defaults to 65536.
The executable requires one positive input size divisible by 16.
Seed 7, hidden/output sizes 16/1, one iteration, CUDA block geometry 16x16,
and the original 1e-3 weight check are preserved. DUMP=yes retains the
baseline input/weight output for external comparison.

Both device kernels retain the CUDA indexing and reduction order. A separate
DMem weight scratch preserves the original restore-before-adjustment behavior.
Momentum scratch is also separate so the original host reference receives its
unchanged input. The CPU neural-network helpers and reference are retained
locally from CUDA. GPU tasks require an enumerated HIP GPU, report the selected
device, and check task submission errors. Nonfinite weights fail verification.
Zero/negative sizes are rejected to avoid the baseline's invalid zero grid.

kernel.cu is provided for future CUDA use; only HIP is validated here.
