# BabelStream IRIS

Uses IRIS DMem and synchronous tasks on an AMD GPU. HIP code objects are loaded
from kernel.hip in the working directory; run with IRIS_ARCHS=HIP.

Defaults and arguments match CUDA: 33554432 elements, 100 iterations,
--arraysize/-s (positive multiple of 256), --numtimes/-n (at least 2),
--help/-h. Both float and double execute all six operations. Initialization is
deterministic (0.1, 0.2, 0.0); the scalar is 0.4. The original pre-timing
dot and nstream reference checks and tolerances are retained.

Build: make
Bounded validation: IRIS_ARCHS=HIP make run RUN_ARGS="-s 65536 -n 3"
GPU executions must be serialized with flock /tmp/hecbench-iris-validation.lock.
The dot reduction retains 256 blocks of 256 threads and flushes partial sums.
Other timed operations leave arrays on device. Timings include IRIS submission
and synchronization; results are based on a variant of BabelStream.
Optional kernel.ptx is unvalidated on this AMD machine.
