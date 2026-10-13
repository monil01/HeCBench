IRIS dmem marchingCubes preserving original analytic field, fixed N1024, isoValue
-0.9, lookup tables, min/max filtering, compaction and triangle-generation kernels.
Native HIP kernel bodies retained, exported as IRIS symbols; counters/aggregate
buffers reset with explicit dmem.update on every original repetition. Atomic
partial writes use rw; output counters use explicit flush_out tasks. Exact float,
unsigned32/64 and uchar4 ABIs retained. Original simulated rendering accumulates
into scalar device buffers, not invented triangle artifacts. Original count-based
PASS verdict and repeat100 default retained. Native original expected counts and
IRIS must both PASS on actual HIP backend3/type256 AMD MI100. CUDA source retained,
NVIDIA unvalidated. No vendor host operations or CPU algorithm fallback.
