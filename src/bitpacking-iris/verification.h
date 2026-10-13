#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>

inline int32_t unpack(const void* data, unsigned bits, int32_t minimum, size_t i) {
  if (!bits) return minimum;
  const auto* bytes = static_cast<const uint8_t*>(data);
  size_t start = i * bits / 8, end = ((i+1)*bits-1)/8;
  uint64_t value = 0;
  for (size_t j = start; j <= end; ++j) value |= uint64_t(bytes[j]) << (8*(j-start));
  return int32_t((value >> ((i*bits)%8)) & ((uint64_t(1)<<bits)-1)) + minimum;
}

inline bool verify(const std::vector<int32_t>& source, const std::vector<int32_t>& input,
                   const void* output, size_t n, int bits, int32_t minimum) {
  std::vector<int32_t> unpacked(n);
  for (size_t i = 0; i < n; ++i) unpacked[i] = unpack(output, bits, minimum, i);
  bool ok = bits <= 13;
  // Preserve the CUDA sampled check, then also check all bounded-size entries.
  for (size_t i = 0; i < size_t(std::sqrt(n))+1; ++i) {
    size_t idx = uint32_t(source[i]) % n;
    ok &= unpacked[idx] == input[idx];
  }
  if (n <= 872163) for (size_t i = 0; i < n; ++i) ok &= unpacked[i] == input[i];
  uint64_t hash = 14695981039346656037ULL;
  const auto* bytes = static_cast<const uint8_t*>(output);
  for (size_t i = 0; i < (n*bits+7)/8; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  printf("CHECK n=%zu min=%d bits=%d packed_hash=%016llx\n", n, minimum, bits,
         static_cast<unsigned long long>(hash));
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}
