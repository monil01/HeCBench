# slit IRIS status

Deferred. The CUDA/HIP benchmark is a cuFFT/hipFFT 2D complex FFT workload.
This porting pass is restricted to IRIS C/C++ kernels using `iris::DMem`/dmem
only, and should not replace the host model with vendor FFT library dispatch.

The native HIP baseline was built and run for evidence; it prints `PASS` for
the default `1024 1000` run. A real IRIS port would need an explicit dmem-based
2D FFT implementation before it can be validated against that baseline.
