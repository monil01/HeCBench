// Device computation is the original RCCL all-reduce, dispatched through
// IRIS_KERNEL_HOST2HIP with DMem-resolved GPU pointers and the IRIS stream.
// See main.cpp. This is a host2hip wrapper library, not a HIP code object;
// IRIS owns all GPU allocations, data movement, task scheduling, and streams.
#include "../../tools/iris_vendor_bridge.cpp"
