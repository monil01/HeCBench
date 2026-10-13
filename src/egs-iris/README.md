# EGS IRIS dmem port

Retains original photon transport, MTGP3217 RNG, material interpolation,
geometry, scattering, counters and detector summation kernel bodies. HIP
backend retains original native HIP vote operations; CUDA backend retains
original CUDA sync-mask operations. No host vendor solver or RNG replacement.

All original device allocations are iris::DMem-backed host storage; no native
allocation/copy/launch APIs. Small exported GPU kernels receive dmem pointers
and bind original device-global pointer tables/geometry metadata. Original
constant-pointer declarations become writable device globals only to permit
binding; transport math/algorithms unchanged.
Geometry metadata uses explicit48/28/12byte scalar PODs reconstructed into
original GPU vector types: GCC and HIP detector_t layout differs28vs32bytes.
Each simulation/summing task registers all indirect dmem payloads with
add_hidden_dmem(iris_rw), preserving
input transfers, partial writes and host flush coherence. Tasks pinned to the
validated HIP default device; no cross-device pointer reuse or CPU fallback.

make / make run preserve original100million histories,80blocks,32768iterations
per launch,16logicalwarps/512threads,512x512detector, seed1325631759 and ./data
paths. Recover original data.tar.gz through DVC before normal run.
HIP_ROOT/IRIS_ROOT configurable, defaults /opt/rocm and /tmp/iris.
CUDA not locally validated.

For bounded original-data validation only, compile both host/kernel with
EXTRA_CFLAGS=-DHECBENCH_EGS_VALIDATION:2blocks,64iterations,4096histories target,
32x32detector. Original source stopping logic may overshoot history target.
Same isolated native input adapter, physics/RNG/warps/default branch unchanged.
Validation compares all detector bin/BMP artifacts and simulation counters;
CPU comparisons/input processing outside per-process bounded GPU lease.
Within the opt-in validation build only, HECBENCH_EGS_HISTORIES and
HECBENCH_EGS_DETECTOR_N override bounded history/detector inputs for larger
coverage checks. Default builds ignore these variables and retain original values.
HECBENCH_EGS_DIAGNOSTIC optionally reports GPU pointer/ABI metadata, not enabled
in normal build/run. Geometry POD fields reconstructed, no downstream Ny clamp.
