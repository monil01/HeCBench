# ccl IRIS port status

Deferred.

`ccl` is an MPI plus NCCL/RCCL all-reduce benchmark. Its CUDA/HIP semantics are
the distributed collective itself: each MPI rank contributes a device buffer and
RCCL/NCCL produces the rank-wise sum on every rank.

This cannot be represented as a correctness-preserving IRIS dmem-only task
launch without replacing the collective with non-equivalent host logic or a
benchmark-specific local fill. That would pass only the deterministic one-rank
case and would not preserve the benchmark behavior.

Native HIP baseline evidence was captured in the local porting logs.
