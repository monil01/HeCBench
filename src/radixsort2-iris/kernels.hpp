#include <stdint.h>
template<class T> __device__ unsigned int key_bits(T key) { return (unsigned int)key; }
template<> __device__ unsigned int key_bits<float>(float key) { return __float_as_uint(key); }

extern "C" __global__ void flags_uint(const unsigned int* keys,int* prefix,int n,int bit) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<n) prefix[i]=((key_bits(keys[i])>>bit)&1)==0;
  if(i==n) prefix[i]=0;
}
extern "C" __global__ void scatter_uint(const unsigned int* keys,const unsigned int* values,
       unsigned int* out_keys,unsigned int* out_values,const int* prefix,int n,int bit,int keysOnly) {
  int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
  int destination=((key_bits(keys[i])>>bit)&1)==0?prefix[i]:prefix[n]+i-prefix[i];
  out_keys[destination]=keys[i];
  if(!keysOnly)out_values[destination]=values[i];
}

extern "C" __global__ void flags_float(const float* keys,int* prefix,int n,int bit) {
  int i=blockIdx.x*256+threadIdx.x;
  if(i<n) prefix[i]=((key_bits(keys[i])>>bit)&1)==0;
  if(i==n) prefix[i]=0;
}
extern "C" __global__ void scatter_float(const float* keys,const unsigned int* values,
       float* out_keys,unsigned int* out_values,const int* prefix,int n,int bit,int keysOnly) {
  int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
  int destination=((key_bits(keys[i])>>bit)&1)==0?prefix[i]:prefix[n]+i-prefix[i];
  out_keys[destination]=keys[i];
  if(!keysOnly)out_values[destination]=values[i];
}

extern "C" __global__ void scan_blocks(int* data,int* sums,int n) {
  __shared__ int work[256];
  int local=threadIdx.x,index=blockIdx.x*256+local;
  int original=index<n?data[index]:0;
  work[local]=original;__syncthreads();
  for(int stride=1;stride<256;stride*=2) {
    int term=local>=stride?work[local-stride]:0;
    __syncthreads();work[local]+=term;__syncthreads();
  }
  if(index<n)data[index]=work[local]-original;
  if(local==255)sums[blockIdx.x]=work[local];
}
extern "C" __global__ void scan_sums(int* sums,int n) {
  if(blockIdx.x||threadIdx.x)return;
  int sum=0;for(int i=0;i<n;++i){int value=sums[i];sums[i]=sum;sum+=value;}
}
extern "C" __global__ void scan_add(int* data,const int* sums,int n) {
  int i=blockIdx.x*256+threadIdx.x;if(i<n)data[i]+=sums[blockIdx.x];
}
