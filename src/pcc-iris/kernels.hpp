
extern "C" __global__ void gram(const float* __restrict__ data,
     float* __restrict__ cormat, int rows,int remaining,int L,int base) {
  size_t index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=size_t(rows)*remaining) return;
  int i=index%rows, j=index/rows;
  float sum=0;
  for(int l=0;l<L;++l) sum+=data[size_t(base+i)*L+l]*data[size_t(base+j)*L+l];
  cormat[index]=sum;
}
extern "C" __global__ void ker(const float * cormat, float * upper,int n)
{
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < (size_t)n * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      //printf("(%lu, %lu) %lu %lu\n", i, j, t, i*n+j);
      upper[t]=cormat[j*n+i];
    }
  }
}

extern "C" __global__ void ker2(const float * cormat, float * upper, int n1, int n)
{
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < (size_t)n1 * n) {
    size_t i = idx/n;
    size_t j = idx%n;
    if(i<j && i<n1)
    {
      size_t t = n * i - i * (i+1) / 2 + j - i - 1;
      upper[t]=cormat[j*n1+i];
    }
  }
}
