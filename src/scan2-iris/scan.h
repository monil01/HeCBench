#ifndef IRIS_SCAN2_SCAN_H
#define IRIS_SCAN2_SCAN_H

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define GROUP_SIZE 256

template <typename T>
int isPowerOf2(T val)
{
  long long v = val;
  return ((v & (-v)) - v == 0 && v != 0) ? 1 : 0;
}

template <typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for (int i = 0; i < bytes; i++) val |= val >> (1 << i);
  val++;
  return val;
}

template <typename T>
int fillRandom(T* arrayPtr, int width, int height, T rangeMin, T rangeMax, unsigned int seed = 123)
{
  if (!arrayPtr) {
    std::cout << "Cannot fill array. NULL pointer." << std::endl;
    return -1;
  }
  if (!seed) seed = static_cast<unsigned int>(time(nullptr));
  srand(seed);
  double range = double(rangeMax - rangeMin) + 1.0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      arrayPtr[index] = rangeMin + T(range * rand() / (RAND_MAX + 1.0));
    }
  }
  return 0;
}

template <typename T>
bool compare(const T* refData, const T* data, int length, T epsilon = 1e-6)
{
  T error = 0.0;
  T ref = 0.0;
  for (int i = 1; i < length; ++i) {
    T diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  T normRef = std::sqrt(ref);
  if (std::fabs(ref) < epsilon) return false;
  T normError = std::sqrt(error);
  error = normError / normRef;
  return error < epsilon;
}

#endif
