# Historical deferral — resolved

The earlier scope restriction below is superseded by explicit authorization
for vendor operations through the IRIS DMem bridge. The complete hipRAND
MRG32K3A port now builds and passes full native-output verification, including
the original default workload. See README.md for current behavior and evidence.
The original note is retained below as historical context, not current status.

# Deferral Details

This benchmark is deferred for the IRIS dmem pass because the CUDA/HIP sources
benchmark cuRAND/hipRAND MRG32K3A generator APIs rather than a translatable
device kernel. A future implementation must reproduce hipRAND-compatible
MRG32K3A output for `HIPRAND_ORDERING_PSEUDO_BEST` before it can be validated
against the native HIP baseline.
