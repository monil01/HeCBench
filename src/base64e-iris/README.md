# base64e IRIS dmem port

This port keeps the CUDA/HIP base64 kernel semantics and uses IRIS `DMem`
objects for host-visible input and output buffers. The native benchmark's
stream-overlap timing structure is represented as repeated IRIS task
submissions while preserving the CPU reference byte comparison as the
correctness check.
