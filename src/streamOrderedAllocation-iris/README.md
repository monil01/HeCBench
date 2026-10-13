# Deferred: streamOrderedAllocation

This directory documents a deferred port, not a working IRIS benchmark.
Status: D1, unsupported allocator semantics in the local IRIS DMem backend.

CUDA compares synchronous allocation, stream-ordered allocation/free and two
memory-pool release thresholds (0 and ULONG_MAX) around synchronization.
The native HIP sibling uses hipMallocAsync/hipFreeAsync and
hipMemPoolSetAttribute. Replacing these modes with ordinary DMem tasks would
erase the allocation policies this benchmark measures.

The installed IRIS source has optional asynchronous allocation internally
(IRIS_MALLOC_ASYNC), but DeviceHIP::MemFree dispatches hipFree; the hipFreeAsync
branch is commented out. Its public DMem/runtime headers provide no memory-pool
release-threshold controls. Faithful implementation requires coordinated IRIS
runtime/API work outside this worker's scope.

No executable, kernels, build or numerical success is claimed for this
assessment. Original CUDA/HIP source snapshots, full native build/run logs,
capability evidence and the manifest/checkpoint are stored under
.porting-state/porting_logs/streamOrderedAllocation-iris/. The native bounded
validation copy changes only nelem/repeat to 4096/2, retaining all allocator
policies and numerical checks; the repository baseline remains untouched.
