#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using uchar = unsigned char;

enum class ScalarKind {
  BFloat16,
  Half,
  Float,
  Int,
  Char,
  UChar
};

struct ConversionCase {
  const char* label;
  const char* kernel;
  ScalarKind dst;
  ScalarKind src;
};

static constexpr ConversionCase kCases[] = {
  {"bfloat16 -> half",  "cvt_bfloat16_half",  ScalarKind::Half,  ScalarKind::BFloat16},
  {"bfloat16 -> float", "cvt_bfloat16_float", ScalarKind::Float, ScalarKind::BFloat16},
  {"bfloat16 -> int",   "cvt_bfloat16_int",   ScalarKind::Int,   ScalarKind::BFloat16},
  {"bfloat16 -> char",  "cvt_bfloat16_char",  ScalarKind::Char,  ScalarKind::BFloat16},
  {"bfloat16 -> uchar", "cvt_bfloat16_uchar", ScalarKind::UChar, ScalarKind::BFloat16},
  {"half -> half",      "cvt_half_half",      ScalarKind::Half,  ScalarKind::Half},
  {"half -> float",     "cvt_half_float",     ScalarKind::Float, ScalarKind::Half},
  {"half -> int",       "cvt_half_int",       ScalarKind::Int,   ScalarKind::Half},
  {"half -> char",      "cvt_half_char",      ScalarKind::Char,  ScalarKind::Half},
  {"half -> uchar",     "cvt_half_uchar",     ScalarKind::UChar, ScalarKind::Half},
  {"float -> float",    "cvt_float_float",    ScalarKind::Float, ScalarKind::Float},
  {"float -> half",     "cvt_float_half",     ScalarKind::Half,  ScalarKind::Float},
  {"float -> int",      "cvt_float_int",      ScalarKind::Int,   ScalarKind::Float},
  {"float -> char",     "cvt_float_char",     ScalarKind::Char,  ScalarKind::Float},
  {"float -> uchar",    "cvt_float_uchar",    ScalarKind::UChar, ScalarKind::Float},
  {"int -> int",        "cvt_int_int",        ScalarKind::Int,   ScalarKind::Int},
  {"int -> float",      "cvt_int_float",      ScalarKind::Float, ScalarKind::Int},
  {"int -> half",       "cvt_int_half",       ScalarKind::Half,  ScalarKind::Int},
  {"int -> char",       "cvt_int_char",       ScalarKind::Char,  ScalarKind::Int},
  {"int -> uchar",      "cvt_int_uchar",      ScalarKind::UChar, ScalarKind::Int},
  {"char -> int",       "cvt_char_int",       ScalarKind::Int,   ScalarKind::Char},
  {"char -> float",     "cvt_char_float",     ScalarKind::Float, ScalarKind::Char},
  {"char -> half",      "cvt_char_half",      ScalarKind::Half,  ScalarKind::Char},
  {"char -> char",      "cvt_char_char",      ScalarKind::Char,  ScalarKind::Char},
  {"char -> uchar",     "cvt_char_uchar",     ScalarKind::UChar, ScalarKind::Char},
  {"uchar -> int",      "cvt_uchar_int",      ScalarKind::Int,   ScalarKind::UChar},
  {"uchar -> float",    "cvt_uchar_float",    ScalarKind::Float, ScalarKind::UChar},
  {"uchar -> half",     "cvt_uchar_half",     ScalarKind::Half,  ScalarKind::UChar},
  {"uchar -> char",     "cvt_uchar_char",     ScalarKind::Char,  ScalarKind::UChar},
  {"uchar -> uchar",    "cvt_uchar_uchar",    ScalarKind::UChar, ScalarKind::UChar},
};

static size_t scalar_size(ScalarKind kind)
{
  switch (kind) {
    case ScalarKind::BFloat16:
    case ScalarKind::Half:
      return sizeof(uint16_t);
    case ScalarKind::Float:
      return sizeof(float);
    case ScalarKind::Int:
      return sizeof(int);
    case ScalarKind::Char:
      return sizeof(char);
    case ScalarKind::UChar:
      return sizeof(uchar);
  }
  return 0;
}

static uint16_t float_to_bfloat16(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

static float bfloat16_to_float(uint16_t value)
{
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t float_to_half(float value)
{
  uint32_t f = 0;
  std::memcpy(&f, &value, sizeof(f));
  const uint32_t sign = (f >> 16) & 0x8000u;
  int exp = static_cast<int>((f >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = f & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant = (mant | 0x800000u) >> (1 - exp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               ((mant + 0x1000u) >> 13));
}

static float half_to_float(uint16_t value)
{
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exp = (value >> 10) & 0x1fu;
  uint32_t mant = value & 0x03ffu;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

template <typename T>
static T clamp_cast(float value)
{
  if constexpr (std::is_same_v<T, char>) {
    return static_cast<char>(static_cast<int>(value));
  } else if constexpr (std::is_same_v<T, uchar>) {
    return static_cast<uchar>(static_cast<int>(value));
  } else {
    return static_cast<T>(value);
  }
}

static float source_value_as_float(const std::vector<unsigned char>& src,
                                   ScalarKind kind,
                                   int index)
{
  switch (kind) {
    case ScalarKind::BFloat16: {
      uint16_t v = 0;
      std::memcpy(&v, src.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
      return bfloat16_to_float(v);
    }
    case ScalarKind::Half: {
      uint16_t v = 0;
      std::memcpy(&v, src.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
      return half_to_float(v);
    }
    case ScalarKind::Float: {
      float v = 0.0f;
      std::memcpy(&v, src.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
      return v;
    }
    case ScalarKind::Int: {
      int v = 0;
      std::memcpy(&v, src.data() + static_cast<size_t>(index) * sizeof(v), sizeof(v));
      return static_cast<float>(v);
    }
    case ScalarKind::Char: {
      char v = 0;
      std::memcpy(&v, src.data() + static_cast<size_t>(index), sizeof(v));
      return static_cast<float>(v);
    }
    case ScalarKind::UChar: {
      uchar v = 0;
      std::memcpy(&v, src.data() + static_cast<size_t>(index), sizeof(v));
      return static_cast<float>(v);
    }
  }
  return 0.0f;
}

template <typename T>
static void store_value(std::vector<unsigned char>& data, int index, T value)
{
  std::memcpy(data.data() + static_cast<size_t>(index) * sizeof(T), &value, sizeof(T));
}

static void initialize_source(std::vector<unsigned char>& src, ScalarKind kind, int nelems)
{
  for (int i = 0; i < nelems; ++i) {
    const float value = static_cast<float>(i % 64);
    switch (kind) {
      case ScalarKind::BFloat16:
        store_value(src, i, float_to_bfloat16(value));
        break;
      case ScalarKind::Half:
        store_value(src, i, float_to_half(value));
        break;
      case ScalarKind::Float:
        store_value(src, i, value);
        break;
      case ScalarKind::Int:
        store_value(src, i, static_cast<int>(value));
        break;
      case ScalarKind::Char:
        store_value(src, i, static_cast<char>(static_cast<int>(value)));
        break;
      case ScalarKind::UChar:
        store_value(src, i, static_cast<uchar>(static_cast<int>(value)));
        break;
    }
  }
}

static bool value_matches(const std::vector<unsigned char>& dst,
                          ScalarKind kind,
                          int index,
                          float ref)
{
  switch (kind) {
    case ScalarKind::Half: {
      uint16_t actual = 0;
      std::memcpy(&actual, dst.data() + static_cast<size_t>(index) * sizeof(actual), sizeof(actual));
      return actual == float_to_half(ref);
    }
    case ScalarKind::Float: {
      float actual = 0.0f;
      std::memcpy(&actual, dst.data() + static_cast<size_t>(index) * sizeof(actual), sizeof(actual));
      return std::fabs(actual - ref) <= 0.0f;
    }
    case ScalarKind::Int: {
      int actual = 0;
      std::memcpy(&actual, dst.data() + static_cast<size_t>(index) * sizeof(actual), sizeof(actual));
      return actual == clamp_cast<int>(ref);
    }
    case ScalarKind::Char: {
      char actual = 0;
      std::memcpy(&actual, dst.data() + static_cast<size_t>(index), sizeof(actual));
      return actual == clamp_cast<char>(ref);
    }
    case ScalarKind::UChar: {
      uchar actual = 0;
      std::memcpy(&actual, dst.data() + static_cast<size_t>(index), sizeof(actual));
      return actual == clamp_cast<uchar>(ref);
    }
    case ScalarKind::BFloat16:
      return false;
  }
  return false;
}

static bool verify_output(const std::vector<unsigned char>& src,
                          const std::vector<unsigned char>& dst,
                          const ConversionCase& conv,
                          int nelems)
{
  for (int i = 0; i < nelems; ++i) {
    const float ref = source_value_as_float(src, conv.src, i);
    if (!value_matches(dst, conv.dst, i, ref)) {
      return false;
    }
  }
  return true;
}

static bool convert_case(const ConversionCase& conv, int nelems, int niters)
{
  std::vector<unsigned char> src(static_cast<size_t>(nelems) * scalar_size(conv.src));
  std::vector<unsigned char> dst(static_cast<size_t>(nelems) * scalar_size(conv.dst));
  initialize_source(src, conv.src, nelems);
  std::fill(dst.begin(), dst.end(), 0);

  iris::DMem mem_dst(dst.data(), dst.size());
  iris::DMem mem_src(src.data(), src.size());

  const int ls = std::min(nelems, 256);
  const size_t global = static_cast<size_t>((nelems + ls - 1) / ls * ls);
  size_t gws[1] = {global};
  size_t lws[1] = {static_cast<size_t>(ls)};

  auto launch = [&]() {
    iris::Task task;
    void* params[3] = {&mem_dst, &mem_src, &nelems};
    int pinfo[3] = {iris_w, iris_r, sizeof(nelems)};
    task.kernel(conv.kernel, 1, nullptr, gws, lws, 3, params, pinfo);
    task.flush_out(mem_dst);
    task.submit(iris_gpu, nullptr, 1);
  };

  launch();

  const auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; ++i) {
    launch();
  }
  const auto end = std::chrono::high_resolution_clock::now();

  mem_dst.fetch(dst.data());
  const double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                      static_cast<double>(niters) / 1.0e6;
  const double size = static_cast<double>(scalar_size(conv.dst) + scalar_size(conv.src)) *
                      static_cast<double>(nelems) / 1.0e9;
  std::printf("size(GB):%.2f, average time(sec):%f, BW:%f\n", size, time, size / time);

  const bool ok = verify_output(src, dst, conv, nelems);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int nelems = std::atoi(argv[1]);
  const int niters = std::atoi(argv[2]);
  if (nelems <= 0 || niters <= 0) {
    std::fprintf(stderr, "number of elements and repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  bool all_ok = true;
  for (const ConversionCase& conv : kCases) {
    std::printf("%s\n", conv.label);
    all_ok = convert_case(conv, nelems, niters) && all_ok;
  }

  platform.finalize();
  return all_ok ? 0 : 2;
}
