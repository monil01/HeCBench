#include <cuda_runtime.h>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

__device__ unsigned int mixRemainder(unsigned int a,
                                     unsigned int b,
                                     unsigned int c,
                                     unsigned int k0,
                                     unsigned int k1,
                                     unsigned int k2,
                                     unsigned int length)
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;
  }

  final(a,b,c);
  return c;
}

extern "C" __global__ void jk3_hash_kernel(
    const unsigned int *__restrict__ lengths,
    const unsigned int *__restrict__ initvals,
    const unsigned int *__restrict__ keys,
    unsigned int *__restrict__ out,
    const int N)
{
  int id = blockDim.x*blockIdx.x+threadIdx.x;
  if (id >= N) return;
  unsigned int length = lengths[id];
  const unsigned int initval = initvals[id];
  auto k = (const uint3*) (keys+id*16);

  unsigned int a,b,c;
  a = b = c = 0xdeadbeef + length + initval;

  uint3 v;
  while (length > 12) {
    v = *k++;
    a += v.x;
    b += v.y;
    c += v.z;
    mix(a,b,c);
    length -= 12;
  }
  v = *k;

  out[id] = mixRemainder(a, b, c, v.x, v.y, v.z, length);
}
