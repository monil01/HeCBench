# Complete three-path dot product with IRIS DMem / HIP

Both float/double suites retain seed19937, uniform integers[-32,32], original
padding/grid/vectorized4-element indexing, 1024-thread hipCUB BlockReduce,
atomic accumulation, 100warmups per path, CLI and default268435456/1000.
No measured implementation is omitted or replaced by CPU computation:

1. Original source-owned hipCUB kernel, as an IRIS HIP code-object task.
2. Genuine hipblasDotEx through the shared IRIS vendor bridge; device-pointer
   mode writes the scalar into DMem and task.flush_out retrieves it.
3. Original GPU stdpar backend through that bridge, explicitly bound to the
   IRIS stream. Installed ROCm hipstdpar numeric.hpp:294-299 forwards the
   original std::transform_reduce(par_unseq,...) to thrust::inner_product
   on the device. We call that SAME GPU primitive with thrust::hip::par.on
   the IRIS stream, avoiding an unselectable default stream. This is a P1
   dispatch-policy adaptation, not a claim that the std C++ policy entrypoint
   itself runs in the port. There is no handwritten replacement reduction
   or CPU fallback in this path. The primitive returns a host scalar exactly
   as the original API does; only its input arrays are device pointers,
   resolved from DMem. Library-internal workspace/copy behavior is retained.

Each result is checked against the original strict T-accumulated reference
and an additional independent int64 sum of all actual input products. The
native diagnostic copy only adds complete scalar-output printing; native
build still uses --hipstdpar on gfx908. DotEx and stdpar destinations/timers
are set to fresh sentinels, so skipped phases cannot pass on stale values.

P1 timings: synchronous IRIS tasks and DMem scalar reset/flush add overhead
to the custom path; stdpar stream policy differs as described above. Every
path retains its own100warmups, repeat loop and timing. Whole-process wall
timings and generated-source/history evidence are recorded locally. Large
FP32 inputs can be reduction-order sensitive; no production-default parity
or performance claim is inferred from bounded correctness tests.

Final sentinel-protected source validated65537elements/repeat3 and17/repeat1:
all12 scalar outputs are exact native/IRIS/int64, with all6 original verdicts
PASS in each case. Earlier4096/repeat2 passed before sentinel hardening and
its separate source/log snapshot is retained. The unchanged native diagnostic
binary uses its recorded30.76s build; final IRIS source is rebuilt separately.
