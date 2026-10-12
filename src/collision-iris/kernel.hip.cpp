#include <hip/hip_runtime.h>

#define WARP_SIZE 32

template <typename K, typename V>
struct Pair {
  __host__ __device__ __forceinline__ Pair() {}
  __host__ __device__ __forceinline__ Pair(K key, V value) : k(key), v(value) {}
  __host__ __device__ __forceinline__ bool operator==(const Pair<K, V>& rhs) const {
    return (k == rhs.k) && (v == rhs.v);
  }
  __host__ __device__ __forceinline__ bool operator<(const Pair<K, V>& rhs) const {
    return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
  }
  K k;
  V v;
};

__device__ __forceinline__ int getBit(int val, int pos) { return (val >> pos) & 0x1; }
__device__ __forceinline__ int getLaneId() { return threadIdx.x % WARP_SIZE; }

template <typename T>
struct LessThan {
  static __device__ __forceinline__ bool compare(const T lhs, const T rhs) {
    return lhs < rhs;
  }
};

template <typename T>
__device__ __forceinline__ T shfl_xor(const T val, int laneMask, int width = WARP_SIZE) {
  return __shfl_xor(val, laneMask, width);
}

template <typename K, typename V>
__device__ __forceinline__ Pair<K, V> shfl_xor(const Pair<K, V>& p, int laneMask, int width = WARP_SIZE) {
  return Pair<K, V>(__shfl_xor(p.k, laneMask, width), __shfl_xor(p.v, laneMask, width));
}

template <typename T, typename Comparator>
__device__ __forceinline__ T shflSwap(const T x, int mask, int dir) {
  T y = shfl_xor(x, mask);
  return Comparator::compare(x, y) == dir ? y : x;
}

template <typename T, typename Comparator>
__device__ T warpBitonicSort(T val) {
  const int laneId = getLaneId();
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 1) ^ getBit(laneId, 0));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 2) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 2) ^ getBit(laneId, 0));
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 3) ^ getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 3) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 3) ^ getBit(laneId, 0));
  val = shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 4) ^ getBit(laneId, 3));
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 4) ^ getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 4) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 4) ^ getBit(laneId, 0));
  val = shflSwap<T, Comparator>(val, 0x10, getBit(laneId, 4));
  val = shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 3));
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 0));
  return val;
}

template <typename T>
__device__ __forceinline__ bool warpHasCollision(T val) {
  val = warpBitonicSort<T, LessThan<T>>(val);
  const T lower = __shfl_up(val, 1);
  const bool dup = (lower == val) && (getLaneId() != 0);
  return (__any(dup) != 0);
}

template <typename T>
__device__ __forceinline__ unsigned int warpCollisionMask(T val) {
  Pair<T, int> pVal(val, getLaneId());
  pVal = warpBitonicSort<Pair<T, int>, LessThan<Pair<T, int>>>(pVal);
  const unsigned long lower = __shfl_up(pVal.k, 1);
  Pair<int, bool> dup(pVal.v, (lower == pVal.k) && (getLaneId() != 0));
  dup = warpBitonicSort<Pair<int, bool>, LessThan<Pair<int, bool>>>(dup);
  return __ballot(dup.v);
}

extern "C" __global__ void checkDuplicates(int num, const int* v, int* hasDuplicate) {
  hasDuplicate[threadIdx.x] = static_cast<int>(warpHasCollision(v[threadIdx.x]));
}

extern "C" __global__ void checkDuplicateMask(int num, const int* v, unsigned int* duplicateMask) {
  unsigned int mask = warpCollisionMask(v[threadIdx.x]);
  if (threadIdx.x == 0) *duplicateMask = mask;
}
