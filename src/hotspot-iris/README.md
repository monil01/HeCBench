IRIS DMem port preserving the CUDA tiled temperature kernel, file arguments,
default 512x512 / pyramid-height 2 / 200 iterations, and native 1e-3 reference
check. All tasks submit synchronously to one AMD GPU; the final ping-pong
buffer is flushed before comparison and file output. No host changes affect
device-resident intermediates between stages.

Build with `make`; run with `IRIS_ARCHS=HIP make run`. Override `RUN_ARGS` for
bounded validation. `IRIS_ROOT`, `IRIS_BUILD`, `CXX`, and `HIPCC` are configurable.
Optional CUDA code-object generation is provided but has not been validated.
Validation evidence is under `.porting-state/porting_logs/hotspot-iris/`.
