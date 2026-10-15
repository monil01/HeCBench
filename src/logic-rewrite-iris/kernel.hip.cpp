// Original graph kernels are compiled into liblogic.so. Each is launched by
// iris_vendor_kernel_launch inside an IRIS task with DMem dependency bindings.
// Vendor Thrust operations use the same task's IRIS-owned HIP stream.
#include "../../tools/iris_vendor_bridge.cpp"
