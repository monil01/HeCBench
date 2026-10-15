# ccl — IRIS DMem and RCCL

This is the original distributed GPU all-reduce, not a local fill or CPU
reduction. IRIS owns the send/receive DMem allocations and transfers. Its
host2hip wrapper resolves GPU pointers and invokes RCCL on the IRIS stream.
MPI supplies rank discovery and broadcasts the RCCL communicator ID.

```sh
make
make run RUN_ARGS=50
```

`IRIS_ROOT`, `IRIS_BUILD`, `ROCM_PATH`, `MPI_ROOT`, `CXX`, and `LAUNCHER` are
configurable. The original CLI (`<repeat>`), all-ones inputs, and size sweep
(1M, 10M, 100M, 1000M float elements) remain the production defaults.

Two optional validation controls also exist in the CUDA/HIP references:

```sh
HECBENCH_CCL_MAX_ELEMENTS=1048576 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 make run RUN_ARGS=2
HECBENCH_CCL_MAX_ELEMENTS=1048576 HECBENCH_CCL_PATTERNED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 make run RUN_ARGS=2
```

Pattern mode uses rank-dependent, signed, nonuniform inputs and checks every
element against its exact rank-wise sum. It is opt-in and does not change the
default workload. Both all-ones and pattern cases passed on two MI100 GPUs,
with 1,048,576 outputs checked per rank in both native HIP and IRIS. The full
1000M-element production sweep has not been validated.

The transport settings disable unavailable peer/IB paths on this machine;
RCCL still performs the GPU collective using its supported transport. They
must match between reference and IRIS tests. IRIS timings include DMem/task
overhead and should not be interpreted as pure RCCL transfer latency.

The actual backend wrapper is `kernel.hip.cpp`, compiled as a shared host2hip
library rather than a HIP code object. See
[bridge documentation](../../tools/iris_vendor_bridge.md). The bridge fails
loudly if a callback fails, because the reference runtime discards wrapper
return codes. No native benchmark GPU allocation, copy, or fill replaces DMem.
