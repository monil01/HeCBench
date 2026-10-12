#include <cmath>
#include <cstdint>
#include <cstring>

template <typename T> static float reference_to_float(T value) {
  return static_cast<float>(value);
}

template <typename T> static T reference_from_float(float value) {
  return static_cast<T>(value);
}

template <> float reference_to_float<HostBFloat16>(HostBFloat16 value) {
  uint32_t bits = static_cast<uint32_t>(value.bits) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <> HostBFloat16 reference_from_float<HostBFloat16>(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return HostBFloat16{static_cast<uint16_t>(bits >> 16)};
}

template <typename T>
void reference(T* out, const T* input, const T* bias, const T* gamma, const T* beta,
               float layernorm_eps, int num_rows, int n) {
  for (int row = 0; row < num_rows; ++row) {
    const int base = row * n;
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) {
      out[base + i] = reference_from_float<T>(reference_to_float(out[base + i]) +
                                              reference_to_float(input[base + i]) +
                                              reference_to_float(bias[i]));
      mean += reference_to_float(out[base + i]);
    }
    mean /= static_cast<float>(n);
    float variance = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float diff = reference_to_float(out[base + i]) - mean;
      variance += diff * diff;
    }
    const float inv_std = 1.0f / std::sqrt(variance / static_cast<float>(n) + layernorm_eps);
    for (int i = 0; i < n; ++i)
      out[base + i] = reference_from_float<T>((reference_to_float(out[base + i]) - mean) *
                                              inv_std * reference_to_float(gamma[i]) +
                                              reference_to_float(beta[i]));
  }
}
