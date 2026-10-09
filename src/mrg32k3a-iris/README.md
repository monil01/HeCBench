# mrg32k3a IRIS status

Deferred for the IRIS dmem pass.

The CUDA/HIP benchmark is a cuRAND/hipRAND API benchmark. Device output is
produced by `curandGenerateUniform` / `hiprandGenerateUniform` for
MRG32K3A with `*_ORDERING_PSEUDO_BEST`, then compared with the corresponding
host generator output. There is no CUDA/HIP kernel body in the source to map
to an IRIS dmem task, and an equivalent IRIS port would need to reproduce the
hipRAND MRG32K3A ordering and skipahead semantics exactly.

Using hipRAND directly from the IRIS host would not be an IRIS dmem port, and
a non-equivalent handwritten RNG kernel would not satisfy the native HIP
validation requirement.
