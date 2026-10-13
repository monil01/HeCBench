All ten CUDA Match algorithms and block geometries use IRIS dmem on HIP.
The initial C rand input stream is generated before IRIS initialization.
Default NPTS=16384, NDIM=128 and run repeat=100 are preserved.
VALIDATION_NPTS=256 bounds deterministic local checks without changing the
algorithm or default. The isolated native adapter uses the same point count
and adds complete index/score dumps. Every kernel is checked against CPU
indices and scores; all native/IRIS artifacts are compared in full.
HECBENCH_ARTIFACT is an optional binary output path (ten index/score pairs).
The inherited kernel header's native main is excluded unless
IRIS_NATIVE_REFERENCE is defined; the actual host is main.cpp using dmem.
