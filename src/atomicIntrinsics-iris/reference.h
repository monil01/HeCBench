#pragma once

#include <cstdio>

template <class T>
static inline T ref_min(T a, T b)
{
  return a < b ? a : b;
}

template <class T>
static inline T ref_max(T a, T b)
{
  return a > b ? a : b;
}

template <class T>
void computeGold(T *gpuData, const size_t len)
{
  T val = 0;
  bool ok = true;

  for (size_t i = 0; i < len; ++i) {
    val += (T)10;
  }

  if (val != gpuData[0]) {
    std::printf("Add failed: %d != %d\n", val, gpuData[0]);
    ok = false;
  }

  val = 0;
  for (size_t i = 0; i < len; ++i) {
    val -= (T)10;
  }

  if (val != gpuData[1]) {
    std::printf("Sub failed: %d != %d\n", val, gpuData[1]);
    ok = false;
  }

  val = (T)(-256);
  for (size_t i = 0; i < len; ++i) {
    val = ref_max(val, (T)i);
  }

  if (val != gpuData[2]) {
    std::printf("Max failed: %d != %d\n", val, gpuData[2]);
    ok = false;
  }

  val = (T)256;
  for (size_t i = 0; i < len; ++i) {
    val = ref_min(val, (T)i);
  }

  if (val != gpuData[3]) {
    std::printf("Min failed: %d != %d\n", val, gpuData[3]);
    ok = false;
  }

  val = 0xff;
  for (size_t i = 0; i < len; ++i) {
    val &= (T)(2 * i + 7);
  }

  if (val != gpuData[4]) {
    std::printf("And failed: %d != %d\n", val, gpuData[4]);
    ok = false;
  }

  val = 0;
  for (size_t i = 0; i < len; ++i) {
    val |= (T)(1 << i);
  }

  if (val != gpuData[5]) {
    std::printf("Or failed: %d != %d\n", val, gpuData[5]);
    ok = false;
  }

  val = 0xff;
  for (size_t i = 0; i < len; ++i) {
    val ^= (T)i;
  }

  if (val != gpuData[6]) {
    std::printf("Xor failed: %d != %d\n", val, gpuData[6]);
    ok = false;
  }

  T limit = 17;
  val = 0;
  for (size_t i = 0; i < len; ++i) {
    val = (val >= limit) ? 0 : val + 1;
  }

  if (val != gpuData[7]) {
    std::printf("atomicInc failed: %d != %d\n", val, gpuData[7]);
    ok = false;
  }

  limit = 137;
  val = 0;
  for (size_t i = 0; i < len; ++i) {
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  }

  if (val != gpuData[8]) {
    std::printf("atomicDec failed: %d != %d\n", val, gpuData[8]);
    ok = false;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
}
