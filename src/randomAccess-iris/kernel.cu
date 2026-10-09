typedef unsigned long long int u64Int;
typedef long long int s64Int;

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#define NUPDATE (4 * TableSize)

extern "C" __device__ u64Int HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;

#pragma unroll
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--) {
    if ((n >> i) & 1) break;
  }

  ran = 0x2;
  while (i > 0) {
    temp = 0;
#pragma unroll
    for (j = 0; j < 64; j++) {
      if ((ran >> j) & 1) temp ^= m2[j];
    }
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
  }

  return ran;
}

extern "C" __global__ void initTable(u64Int* Table, const u64Int TableSize)
{
  const u64Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < TableSize) Table[i] = i;
}

extern "C" __global__ void update(u64Int* __restrict__ Table, const u64Int TableSize)
{
  const int j = threadIdx.x;
  u64Int ran = HPCC_starts((NUPDATE / 128) * j);
  for (u64Int i = 0; i < NUPDATE / 128; i++) {
    ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
    atomicXor(&Table[ran & (TableSize - 1)], ran);
  }
}
