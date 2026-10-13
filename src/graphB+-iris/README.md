IRIS dmem port retaining graphB+ kernels and original signed-graph/tree/majority
algorithms. HIP backend kernels retain existing native warp-size adaptations;
CUDA kernels retain their original32-lane implementation. Defaults and graph
CSV/iteration/output CLI remain unchanged. HIP runtime calls in the host are
device-property queries only, preserving the hardware-dependent block count;
all device memory ownership/copies/launches use dmem/tasks. Tasks explicitly
target Device0 because the original kernels retain device-global hi/wSize state.

Root-label partial reset fetches/updates the complete array to preserve other
device values. Final nlist/inTree/negCnt are materialized before original output
writing: native source omitted these copies, yielding uninitialized edge stats.
The isolated native validation adapter adds only the same missing output copies,
without changing kernels or defaults. Verification is enabled for bounded
connected signed graphs and compares every node/edge artifact row.
CUDA backend is unvalidated on NVIDIA; IRIS AMD validation requires HIP.

