# Deferral Details

This benchmark is deferred for the IRIS dmem pass because the CUDA/HIP sources
benchmark cuRAND/hipRAND MRG32K3A generator APIs rather than a translatable
device kernel. A future implementation must reproduce hipRAND-compatible
MRG32K3A output for `HIPRAND_ORDERING_PSEUDO_BEST` before it can be validated
against the native HIP baseline.
