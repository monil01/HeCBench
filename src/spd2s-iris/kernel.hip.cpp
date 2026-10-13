#include <hip/hip_runtime.h>
#include <cstdint>
extern "C" __global__ void count_rows(const float* dense,int64_t* counts,int64_t rows,int64_t cols) {
  const int64_t r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=rows)return;
  int64_t count=0;
  for(int64_t c=0;c<cols;++c) count+=dense[r*cols+c]!=0.0f;
  counts[r]=count;
}
extern "C" __global__ void prefix_rows(const int64_t* counts,int64_t* offsets,int64_t rows) {
  if(blockIdx.x||threadIdx.x)return;
  offsets[0]=0;
  for(int64_t r=0;r<rows;++r)offsets[r+1]=offsets[r]+counts[r];
}
extern "C" __global__ void scatter_rows(const float* dense,const int64_t* offsets,int64_t* columns,float* values,int64_t rows,int64_t cols) {
  const int64_t r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=rows)return;
  int64_t pos=offsets[r];
  for(int64_t c=0;c<cols;++c) if(dense[r*cols+c]!=0.0f) {
    columns[pos]=c; values[pos]=dense[r*cols+c]; ++pos;
  }
}
