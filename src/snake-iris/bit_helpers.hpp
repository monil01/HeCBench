#define warp_size 32
#define NBytes 8

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline uint lsr(uint x, int sa) {
  if(sa > 0 && sa < 32) return (x >> sa);
  return x;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline uint lsl(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x << sa);
  return x;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline uint set_bit(uint &data, int y) {
  data |= lsl(1, y);
  return data;
}


