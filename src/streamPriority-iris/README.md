# streamPriority IRIS status

Deferred for this IRIS dmem pass.

The CUDA/HIP benchmark is specifically a stream-priority and event-scheduling
benchmark: it compares default-stream execution with a high-priority training
stream and a low-priority prefetch stream coordinated by events. The current
IRIS dmem porting scope can express the individual kernels and memory buffers,
but it does not expose equivalent native HIP stream-priority controls through
the IRIS task interface. A sequential IRIS translation would validate the math
but would not preserve the benchmark behavior being measured.

Native HIP baseline compilation was checked and recorded in the local porting
logs. A correctness-only IRIS variant should be treated as a different
benchmark unless IRIS adds explicit stream priority and event ordering support
for HIP backend task submission.
