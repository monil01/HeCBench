// Genuine hipSPARSE SDDMM runs in the callback in main.cpp, through DMem pointers.
// This is an IRIS host2hip wrapper shared library, not a HIP device code object.
#include "../../tools/iris_vendor_bridge.cpp"
