# Quartic minimum solver with IRIS DMem

Original QRdel and QuarticSolver kernels are retained. Default N=1999999,
repeat argument, seed, CPU solver, two PASS checks, and 64-thread launch remain.
Host allocations use malloc; all backend buffers use DMem. The stream variant
retains four equal chunks plus residual; synchronous IRIS tasks replace native
stream scheduling (P1 concurrency compromise, algorithm and coverage unchanged).
Optional HECBENCH_QEM_N bounds validation and HECBENCH_QEM_OUTPUT dumps all
minimum values for both paths. Isolated native adapter adds only identical
size/output hooks and missing CUDA CPU reference header. No native changes.
