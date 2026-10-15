# DMem HIP vendor-operation bridge

`iris_vendor_bridge.cpp` implements the existing IRIS host2hip wrapper ABI.
It allows an IRIS task to call a GPU vendor library using pointers resolved
from its DMem arguments and the stream supplied by IRIS. This is not a CPU
fallback and does not allocate, copy, or launch native benchmark kernels.
The callback may call the original GPU library operation on those pointers.

Build it as a shared library, set `IRIS_KERNEL_HOST2HIP` to that library before
IRIS initialization, and dispatch an IRIS kernel named `hecbench_vendor`.
Argument zero is a scalar `hecbench_vendor_callback`; subsequent arguments are
the callback's scalars or DMem objects, with normal `iris_r/w/rw` access modes.
The callback uses `hecbench_vendor_arg<T>(args, index)` to read an argument.
It receives a HIP stream handle as `void*` and the IRIS device index.

The bridge supports 24 slots of at most 32 bytes, within the 1024-byte wrapper
storage of reference IRIS revision `ff57e620a92fcf62eff3568d9c6eeaaf078f0340`.
Callbacks are in-process C++ function pointers, not serialized data. Keep all
callback state alive until task completion. Do not throw across the C ABI.
The callback must return zero on success and ensure the operation is finished
before returning unless the task/runtime's asynchronous contract is used.

The reference loader ignores callback return codes. Consequently the bridge
aborts on any callback error, missing callback, or invalid argument rather than
silently accepting a failed operation. There is no global mutable dispatch
state: argument storage is per IRIS kernel wrapper.

`src/ccl-iris` demonstrates a real multi-process RCCL all-reduce with full-array
validation. DMem owns both GPU buffers; MPI broadcasts only the RCCL unique ID.
