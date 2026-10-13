IRIS dmem port preserving all six original resnet modes, tensor dimensions,
launch geometry, Winograd and 1x1 algorithms, file paths, and default repetitions.
Fixed launch-specific shared-memory arrays replace dynamic shared allocations
with the same capacity, because these kernel geometries are constants.
Host arrays use dmem; output and intermediates start zeroed, partial writes use
iris_rw. CUDA kernels are retained for future NVIDIA use.

RESNET_DUMP is an optional validation adapter that appends the complete float
output after each repetition. The isolated native HIP validation copy adds only
the identical output-dump hook; original siblings are untouched. Validation
seeds the original Python generator externally (123), shares the resulting data,
runs all modes three times, and compares every output element including padding.
Default data generation remains the original unseeded generator.

