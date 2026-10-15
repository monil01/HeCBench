# Complete HPL with IRIS DMem / real hipBLAS

This product includes software developed at the University of Tennessee,
Knoxville, Innovative Computing Laboratory. See LICENSE.md and the original
HPL source copyright notices.

The complete original130-file HPL2.3 CPU/MPI solver is built from HPL_ROOT
(default ../hpl-cuda/src/hpl-2.3), not replaced with a toy GEMM/solver. Only
the original GPU DGEMM/DTRSM shim is ported: DMem owns its GPU buffers,
IRIS tasks resolve their pointers, and the shared bridge invokes the original
real hipBLAS operations on an IRIS-managed stream. All input shapes, leading
dimensions, alpha/beta, fixed NoTrans/Left/Lower/Unit settings and thresholds
follow the original shim. Its intentionally hybrid small-size CBLAS branches
are retained unchanged; no new CPU/backend fallback exists. Validation
requires BOTH actual GPU DGEMM and DTRSM calls, not merely HPL PASSED output.

The HPL.dat interface, original deterministic matrix generation, default
HPL_small_gpu.dat and original residual threshold16.0 remain. make run copies
that original deck into .run. HPL_INPUT is configurable; validation.HPL.dat
changes ONLY problem sizes to1536/1665 and NB to512, leaving the other original
parameters unchanged. This covers an unaligned tail and both GPU thresholds.
The large production default24576/NB2048 is retained but not claimed tested.

Both native and port use the available system CBLAS-compatible libblas;
BLAS_LIBS is configurable. No global LAPACK clone/install or shared build
configuration is required. make native compiles the UNCHANGED original native
HIP CPU source and GPU shim in .build/native, with a renamed driver entrypoint
and diagnostic-only link wrappers; it does not use replacement GPU arithmetic.
The port's original source core is separately built in .build/hpl.

P1: the port initializes MPI with FUNNELED support before IRIS worker startup;
original driver MPI initialization/finalization are link-wrapped so IRIS can
shut down before MPI finalization. Cleanup runs INSIDE the original driver's
MPI_Finalize hook because the original driver then calls exit(0), not return.
Native diagnostic counters use atexit for that same reason. Original HPL MPI
communication is retained.
Synchronous DMem flushes/task dispatch and releasing per-operation GPU handles
affect performance; no kernel-only performance equivalence is claimed.

observe.c only observes the result after the original complete HPL_pdgesv,
before HPL regenerates its matrix for the usual whole-system residual check.
HECBENCH_HPL_DUMP=/absolute/prefix emits every active LU/RHS element and every
solution element (padding excluded), for every rank/test. vendor_counts.cpp
observes actual hipBLAS calls via linker wrapping. verify.py requires both
full-system PASSED verdicts, matching positive DGEMM/DTRSM GPU call counts,
and finite whole factor/solution agreement at atol=rtol1e-10. These diagnostic
wrappers do not change any benchmark matrix or vendor operation.

Build: make -j4; optional baseline: make native -j4.
Bounded test: make run HPL_INPUT=validation.HPL.dat.
Native: make native-run HPL_INPUT=validation.HPL.dat.
MPICC, HIPCC, IRIS_ROOT, IRIS_BUILD, HPL_ROOT, HPL_NATIVE_ROOT, BLAS_LIBS and
LAUNCHER are configurable. The default MPI launcher is one rank as originally.

Validated N1536 and1665, NB512, one MPI rank. All5137923 active LU/RHS/solution
doubles were BIT-EXACT native/IRIS; both original residual checks passed.
Actual native and IRIS GPU counts were23 DGEMM and2 DTRSM calls each.
