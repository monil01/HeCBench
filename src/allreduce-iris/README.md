# Direct-GPU MPI ring allreduce with IRIS DMem

Preserves the original ring scatter-reduce (Irecv/Send/Wait plus GPU addition)
and pipelined ring allgather (Sendrecv), segment partition including residuals,
original size/iteration lists, ones input and exact world-sum check. No host
array is passed to the payload MPI calls: the vendor bridge resolves DMem-owned
GPU pointers. Each receive is followed by an actual IRIS HIP add kernel with the
original 32x256 grid-stride geometry. Metadata length gathering remains host MPI.
Input-to-output device copying is an IRIS GPU copy task (P1: not native DMA copy);
task/lifetime/allocation overhead differs, so throughput parity is not claimed.

Select the local GPU before MPI initialization. MPI_THREAD_MULTIPLE is required
because bridge callbacks call MPI on IRIS worker threads. The GPU-aware MPI build
and its UCX dependency must be supplied; the system MPI on this validation host
does not support ROCm buffers. Device-pointer attributes are asserted in callbacks.
No shared IRIS runtime changes, native allocation/copy, or CPU reduction fallback.

Example (set GPU_MPI_ROOT to your ROCm-aware OpenMPI installation):
  make MPI_ROOT=$GPU_MPI_ROOT
  IRIS_ARCHS=HIP make run MPI_ROOT=$GPU_MPI_ROOT \
    LAUNCHER="$GPU_MPI_ROOT/bin/mpirun --mca pml ucx --mca pml_ucx_tls any --mca pml_ucx_devices any -n 2"

UCX_TLS=sm,self,rocm_copy,tcp selects the ROCm-copy transport for local validation.
The installed rocm_ipc transport fails native HIP IPC registration on smaller
allocations; it is omitted. MPI receives actual GPU pointers, while its internal
copy/staging policy is supplied by UCX, identically for native and IRIS models.
OpenMPI's default UCX eligibility whitelist otherwise requires IPC/RDMA;
pml_ucx_tls=any and pml_ucx_devices=any explicitly allow the copy transport.
LD_LIBRARY_PATH must include that MPI lib directory, its UCX lib directory and
ROCm lib if their builds do not supply runtime paths. The application does not
replace or force transport policy.

Optional matched CUDA/HIP/IRIS validation hooks (unset means unchanged defaults):
HECBENCH_ALLREDUCE_MAX_ELEMENTS, HECBENCH_ALLREDUCE_MAX_ITERATIONS cap the suite.
HECBENCH_ALLREDUCE_TEST_LENGTH selects one explicit size for uneven/empty chunks.
HECBENCH_ALLREDUCE_PATTERNED uses exact signed index/rank-dependent inputs.
HECBENCH_DUMP_PREFIX exports every rank's complete output after every iteration.
The hooks change no native collective, MPI ordering, GPU kernel or default workload.
Bounded test results do not validate production-default timing/iteration coverage.
CUDA kernel source is provided but no CUDA runtime/backend validation is claimed.
