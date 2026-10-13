DEFERRED: sparkler is not a working IRIS port.

The semantic source uses MPI across four ranks and vendor BLAS SGEMM calls on
native device pointers and streams. The HIP sibling uses hipBLAS. Its default
arguments remain `--num_vector 4000 --num_field 90000 --num_iterations 10` with
four MPI processes. Porting only the initialization kernel would omit the
benchmark's GEMM computation and checksum semantics.

This DMem/kernel-only pass does not provide an IRIS vendor-BLAS task adapter
for those native pointer/stream calls. A follow-up needs explicit library
interop and distributed validation; no substitute GEMM or host fallback is
claimed here. See `.porting-state/porting_logs/sparkler-iris/manifest.json` for
complete snapshots and the native build/dependency evidence.
