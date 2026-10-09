# dwconv1d IRIS status

Deferred.

The CUDA and HIP siblings are PyTorch extension benchmarks (`run.py` JIT-loads
`src/timex_op.cpp` and `src/timex.cu`) rather than standalone native HIP
executables. On this machine, the native HIP validation command
`make -C src/dwconv1d-hip run` fails before benchmark execution because
PyTorch's extension loader requires `CUDA_HOME`.

No IRIS dmem port is committed for this benchmark because the required native
HIP reference result is unavailable and the benchmark's current host interface
is a framework extension, not a direct executable baseline.
