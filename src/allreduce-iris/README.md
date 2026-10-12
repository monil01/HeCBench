# allreduce IRIS port

This port preserves the CUDA ring allreduce's segmenting, MPI neighbor
communication, size/iteration list, and exact `world`-value verification.
IRIS `DMem` owns the reduction output and receive buffer; MPI uses their
host-side synchronized views because the portable IRIS dmem API does not
expose a raw device pointer for CUDA-aware MPI. The HIP kernel performs each
segment reduction and has a bounds guard for the rounded work size.

The default run is unchanged from the CUDA/HIP source. Use `IRIS_ARCHS=HIP`
and an MPI launcher with two ranks.
