IRIS data-memory port of ECL-BH v4.5.

Build with `make`; run with `IRIS_ARCHS=HIP make run`. The default run arguments
remain `100000 100`. Override with `RUN_ARGS="256 2"` for a bounded case.
`IRIS_ROOT`, `IRIS_BUILD`, `HIPCC`, `CXX`, and `LAUNCHER` are configurable.
Build with `DEBUG=yes` to retain the CUDA/HIP per-body verification output.

The host owns arrays through IRIS `DMem`, submits each stage synchronously to
one AMD GPU, and flushes position, acceleration, and velocity outputs. The nine
kernels retain the source block counts (24 with the original stage factors),
256-thread block sizes, tree traversal, and seed 7. Initialization uses one
thread. `kernels.h` shares the kernels between HIP and optional CUDA builds;
the HIP vote primitive matches the native HIP sibling (`__all`), while CUDA
uses `__all_sync`. CUDA has not been validated here.

The original benchmark has no unconditional PASS/FAIL reference check.
Validation compares all nine printed scalar outputs per body against an
unmodified native HIP source snapshot with `DEBUG=yes`, requiring finite
values and rtol 0.005 / atol 1e-7. The relative tolerance accounts for the
source's three-significant-digit output and floating-point tree reductions.
Performance timings include synchronous IRIS submission overhead.

GPU validation must use `flock /tmp/hecbench-iris-validation.lock` with a bounded
execution timeout. Evidence and full attempt history are in
`.porting-state/porting_logs/bh-iris/`; check its manifest for actual outcomes.
