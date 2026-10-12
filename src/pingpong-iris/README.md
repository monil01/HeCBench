# pingpong IRIS status

Deferred.

The CUDA/HIP benchmark validates GPU-aware MPI and NCCL/RCCL ping-pong
transfers using native device pointers. The current IRIS dmem-only porting
scope exposes memory through `iris::DMem` task arguments and does not provide a
portable native device pointer that MPI or RCCL can send and receive directly.

Native HIP evidence was collected before deferring: the MPI target compiled,
but the two-rank run emitted repeated MPI read errors on device-buffer
transfers and was interrupted after 119 seconds while sweeping larger payloads.
A host-buffer MPI rewrite would no longer validate the same CUDA/HIP semantics,
so it is intentionally not counted as a completed IRIS port.
