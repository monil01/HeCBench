#include "md5_common.h"
extern "C" __global__
void md5hash_kernel(
  int *__restrict__ foundIndex,
  unsigned char *__restrict__ foundKey,
  unsigned int *__restrict__ foundDigest,
  int keyspace,
  const int byteLength,
  const int valsPerByte,
  const unsigned int searchDigest0,
  const unsigned int searchDigest1,
  const unsigned int searchDigest2,
  const unsigned int searchDigest3 )
{
  int threadid = blockDim.x * blockIdx.x + threadIdx.x;
  int startindex = threadid * valsPerByte;
  unsigned char key[8] = {0,0,0,0, 0,0,0,0};
  IndexToKey(startindex, byteLength, valsPerByte, key);

  for (int j=0; j < valsPerByte && startindex+j < keyspace; ++j)
  {
    unsigned int digest[4];
    md5_2words((unsigned int*)key, byteLength, digest);
    if (digest[0] == searchDigest0 &&
        digest[1] == searchDigest1 &&
        digest[2] == searchDigest2 &&
        digest[3] == searchDigest3)
    {
      foundIndex[0] = startindex + j;
      foundKey[0] = key[0];
      foundKey[1] = key[1];
      foundKey[2] = key[2];
      foundKey[3] = key[3];
      foundKey[4] = key[4];
      foundKey[5] = key[5];
      foundKey[6] = key[6];
      foundKey[7] = key[7];
      foundDigest[0] = digest[0];
      foundDigest[1] = digest[1];
      foundDigest[2] = digest[2];
      foundDigest[3] = digest[3];
    }
    ++key[0];
  }
}
