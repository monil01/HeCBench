#ifndef HECBENCH_IRIS_VENDOR_BRIDGE_HPP
#define HECBENCH_IRIS_VENDOR_BRIDGE_HPP
#include <cstdint>
#include <cstring>

// An IRIS host2hip callback receives resolved DMem GPU pointers, scalar copies,
// and the IRIS-owned HIP stream. It must not allocate or copy native GPU memory.
// Slot zero contains the callback itself; its ordinary arguments start at one.
using hecbench_vendor_callback = int (*)(void*, unsigned, void* const*);
template <class T> inline T hecbench_vendor_arg(void* const* args, unsigned i) {
  static_assert(sizeof(T) <= 32, "IRIS bridge argument too large");
  T value;
  std::memcpy(&value, args[i], sizeof(T));
  return value;
}
#endif
