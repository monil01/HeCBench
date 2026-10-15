# IRIS DMem GPU pingpong: MPI and genuine RCCL

Builds main-mpi and main-nccl; make run executes BOTH original variants in separate
two-rank MPI jobs. main is a generated alias for main-mpi for generic tooling.
Original CLI (no required args), exponents16..27 (65536..134217728 doubles,
512KiB..1GiB), five warm-up iterations and fifty timed round trips are retained.

GPU payload memory belongs only to IRIS DMem. Existing vendor bridge resolves
device pointers into GPU-aware MPI Send/Recv or genuine RCCL ncclSend/ncclRecv.
During each warm-up, rank1 executes the original +1 double kernel through an
actual IRIS task with original1024x256 grid-stride geometry, then returns data.
The timed transfer-only loop is one bridge task, retaining original MPI blocking
or RCCL queued stream ordering and final stream synchronization. All outputs are
checked after warm-up and timed transfer on both ranks; no CPU increment fallback.
P1: task boundaries/synchronization and host-backing lifetime differ, so native
throughput parity is not claimed. CUDA backend is not validated.

Bind the local GPU before MPI_Init_thread(MPI_THREAD_MULTIPLE), required for MPI
calls on IRIS workers. Native CUDA/HIP siblings gain this minimal initialization
ordering correction, matched optional validation hooks, and HIP Makefile MPI_ROOT
link-path correction. Original untouched snapshots and rationale are retained.

Use a ROCm-aware MPI/UCX build:
 make MPI_ROOT=$GPU_MPI_ROOT
 IRIS_ARCHS=HIP make run MPI_ROOT=$GPU_MPI_ROOT \
   LAUNCHER="$GPU_MPI_ROOT/bin/mpirun --mca pml ucx --mca pml_ucx_tls any --mca pml_ucx_devices any -n 2"
For this host UCX_TLS=sm,self,rocm_copy,tcp avoids native rocm_ipc registration
failures; its internal staging is MPI transport policy, not application host
buffers. LD_LIBRARY_PATH must find that MPI, UCX and ROCm installation.
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 are the locally supported RCCL transport
settings; real GPU RCCL transfers still execute, not a mock or CPU substitute.
Supplementary native testing with HSA_FORCE_FINE_GRAIN_PCIE=1 and
NCCL_P2P_DISABLE=0 still fails RCCL communicator setup: transport/p2p.cc:256,
invalid argument. No IRIS peer-transport parity is claimed from that failure.
Direct peer-access support alone does not establish working RCCL IPC transport;
the passing matched P2P-disabled results remain the validation basis.

Optional matching native/IRIS hooks (unset preserves defaults):
HECBENCH_PINGPONG_MAX_EXP bounds the exponent sweep.
HECBENCH_PINGPONG_MAX_ITERATIONS caps timed round trips (warm-up stays5).
HECBENCH_PINGPONG_TEST_LENGTH selects one explicit size for edge/stride tests.
HECBENCH_PINGPONG_PATTERNED uses signed index/rank inputs instead of zeros.
HECBENCH_DUMP_PREFIX records complete per-rank warm/timed double outputs.
Bounded evidence is not production-default timing/iteration validation.
