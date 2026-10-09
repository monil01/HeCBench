# concurrentKernels IRIS port status

Deferred.

`concurrentKernels` is a CUDA/HIP stream and event concurrency sample. Its core
behavior is launching `nkernels` independent kernels into separate streams,
recording one event per stream, and making a final stream wait for those events
before running the reduction kernel and device-to-host copy.

An IRIS dmem-only port can express the local kernels and buffers, but it cannot
preserve the explicit CUDA/HIP stream/event dependency graph that this benchmark
is designed to measure. Replacing it with sequential task submissions would
preserve the final scalar value but not the benchmark semantics.

Native HIP baseline evidence was captured in the local porting logs.
