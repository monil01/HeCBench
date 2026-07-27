// called by host and device

#pragma omp declare target
inline uchar4 operator^(uchar4 a, uchar4 b)
{
  return {(uchar)(a.x ^ b.x), (uchar)(a.y ^ b.y), 
          (uchar)(a.z ^ b.z), (uchar)(a.w ^ b.w)};
}

inline void operator^=(uchar4 &a, const uchar4 b)
{
  a.x ^= b.x;
  a.y ^= b.y;
  a.z ^= b.z;
  a.w ^= b.w;
}

uchar galoisMultiplication(uchar a, uchar b)
{
    uchar p = 0; 
    for(unsigned int i=0; i < 8; ++i)
    {
        if((b&1) == 1)
        {
            p^=a;
        }
        uchar hiBitSet = (a & 0x80);
        a <<= 1;
        if(hiBitSet == 0x80)
        {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    return p;
}

inline
uchar4 sboxRead(const uchar * SBox, uchar4 block)
{
    return {SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]};
}

uchar4 mixColumns(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
    unsigned int bw = 4;

    uchar x, y, z, w;

    x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
    y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
    z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
    w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);
   
    for(unsigned int k=1; k< 4; ++k)
    {
        x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
        y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
        z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
        w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
    }
    
    return {x, y, z, w};
}

uchar4 shiftRows(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        //r.xyzw() = r.yzwx();
        uchar x = r.x;
        uchar y = r.y;
        uchar z = r.z;
        uchar w = r.w;
        r = {y,z,w,x};
    }
    return r;
}

uchar4 shiftRowsInv(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        // r = r.wxyz();
        uchar x = r.x;
        uchar y = r.y;
        uchar z = r.z;
        uchar w = r.w;
        r = {w,x,y,z};
    }
    return r;
}
#pragma omp end declare target

void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                const uint     width ,
                const uint     height ,
                const uint     rounds )
{
   const unsigned int numBlocks = width*height/16;
   #pragma omp parallel for
   for (unsigned int blk = 0; blk < numBlocks; ++blk) {
     uchar4 block0[4];
     uchar4 block1[4];
     uchar4 galiosCoeff[4];
     galiosCoeff[0] = {2, 0, 0, 0};
     galiosCoeff[1] = {3, 0, 0, 0};
     galiosCoeff[2] = {1, 0, 0, 0};
     galiosCoeff[3] = {1, 0, 0, 0};
     unsigned int bx = blk % (width/4);
     unsigned int by = blk / (width/4);
     unsigned int base = ((by * (width/4)) + bx) * 4;

     for (unsigned int lid = 0; lid < 4; ++lid) {
       block0[lid] = input[base + lid];
       block0[lid] ^= roundKey[lid];
     }
     for (unsigned int r = 1; r < rounds; ++r) {
       for (unsigned int lid = 0; lid < 4; ++lid) {
         block0[lid] = sboxRead(SBox, block0[lid]);
         block0[lid] = shiftRows(block0[lid], lid);
       }
       for (unsigned int lid = 0; lid < 4; ++lid) {
         block1[lid] = mixColumns(block0, galiosCoeff, lid);
       }
       for (unsigned int lid = 0; lid < 4; ++lid) {
         block0[lid] = block1[lid] ^ roundKey[r*4 + lid];
       }
     }
     for (unsigned int lid = 0; lid < 4; ++lid) {
       block0[lid] = sboxRead(SBox, block0[lid]);
       block0[lid] = shiftRows(block0[lid], lid);
       output[base + lid] = block0[lid] ^ roundKey[rounds*4 + lid];
     }
   }
}

void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                const  uint    width ,
                const  uint    height ,
                const  uint    rounds)
{
  const unsigned int numBlocks = width*height/16;
  #pragma omp parallel for
  for (unsigned int blk = 0; blk < numBlocks; ++blk) {
    uchar4 block0[4];
    uchar4 block1[4];
    uchar4 galiosCoeff[4];
    galiosCoeff[0] = {14, 0, 0, 0};
    galiosCoeff[1] = {11, 0, 0, 0};
    galiosCoeff[2] = {13, 0, 0, 0};
    galiosCoeff[3] = { 9, 0, 0, 0};
    unsigned int bx = blk % (width/4);
    unsigned int by = blk / (width/4);
    unsigned int base = ((by * (width/4)) + bx) * 4;

    for (unsigned int lid = 0; lid < 4; ++lid) {
      block0[lid] = input[base + lid];
      block0[lid] ^= roundKey[4*rounds + lid];
    }
    for (unsigned int r = rounds - 1; r > 0; --r) {
      for (unsigned int lid = 0; lid < 4; ++lid) {
        block0[lid] = shiftRowsInv(block0[lid], lid);
        block0[lid] = sboxRead(SBox, block0[lid]);
      }
      for (unsigned int lid = 0; lid < 4; ++lid) {
        block1[lid] = block0[lid] ^ roundKey[r*4 + lid];
      }
      for (unsigned int lid = 0; lid < 4; ++lid) {
        block0[lid] = mixColumns(block1, galiosCoeff, lid);
      }
    }
    for (unsigned int lid = 0; lid < 4; ++lid) {
      block0[lid] = shiftRowsInv(block0[lid], lid);
      block0[lid] = sboxRead(SBox, block0[lid]);
      output[base + lid] = block0[lid] ^ roundKey[lid];
    }
  }
}
