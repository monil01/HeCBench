# Complete ILU-preconditioned BiCGStab via IRIS DMem

Original matrix.mtx/maxIterations/verbose CLI, real symmetric MatrixMarket parser,
B=0.75*A*ones, zero initial guess, double precision, ILU0 lower-unit/upper-nonunit
preconditioner, complete BiCGStab recurrence and tolerance 1e-10 retained.
The original iteration's P-copy ordering is deliberately preserved, not silently
rewritten into another BiCGStab variant. The original NVIDIA notice is in solver.hpp.

Genuine hipSPARSE ILU0, SpMV, SpSV analysis/solve and hipBLAS dot/norm/axpy/scal run
through the committed vendor bridge with DMem-owned CSR arrays, factor values,
vectors, and four workspaces on the IRIS stream. Setup queries workspace sizes;
the second task performs the entire original algorithm. Native device-to-device
copies/zeroing become hipBLAS Dcopy, including copies from an extra DMem zero
vector. No native allocation/copy or CPU fallback performs the solver.
Zero-byte optional SpMV workspace is passed as nullptr, matching native behavior;
passing a non-null placeholder to that API is invalid on the installed hipSPARSE.
The native source's uninitialized alpha in SpSV query/analysis is set to 1; these
structural phases do not depend on alpha. All library errors abort loudly.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS="/path/TEM27623_M.mtx 100 0"`.
The original default path/100-iteration setting is unchanged. Matrix data is not
vendored. Optional HECBENCH_DUMP_PREFIX exports the complete B, X and ILU factors.
An independent long-double CPU residual is diagnostic only, not solver fallback.
P1: task/setup overhead and BLAS copies differ from native runtime copies; no
throughput parity or CUDA backend validation is claimed. CUDA/HIP sources remain
untouched. Validation details and original public dataset provenance are local
under .porting-state/porting_logs/bicgstab-iris.
