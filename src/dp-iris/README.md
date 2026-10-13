The full dp benchmark is deferred in this IRIS dmem-only batch.

CUDA main.cu benchmarks three independent implementations for both float and
double: a CUB block reduction kernel, cublasDotEx, and GPU standard parallel
std::transform_reduce. The HIP sibling uses hipcub, hipblasDotEx, and
--hipstdpar. Porting just the first reduction would omit two measured paths,
their warmups, timings, and correctness checks.

No IRIS task binding for the vendor DotEx or standard parallel execution path
is supplied by this benchmark. Direct vendor launches from the host would
violate this batch's dmem-only requirement. An approved IRIS library binding
or broader semantic replacement is needed before a full faithful port.
The original seed 19937, float/double types, 100 warmup iterations, CLI,
and Makefile workload 268435456 elements / 1000 repeats remain authoritative.

This directory documents a dependency deferral; it contains no executable
port and does not claim IRIS correctness or GPU validation.
