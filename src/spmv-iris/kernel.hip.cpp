#include <hip/hip_runtime.h>
#include "mv.h"
extern "C" __global__ void mv_dense(size_t n,const REAL* matrix,const REAL* x,REAL* y){
 size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=n)return;
 REAL sum=0;for(size_t j=0;j<n;++j)if(matrix[row*n+j]!=REAL(0))sum+=matrix[row*n+j]*x[j];y[row]=sum;
}
extern "C" __global__ void mv_csr(const size_t* offsets,const size_t* columns,const REAL* values,const REAL* x,REAL* y,size_t n,size_t nnz){
 size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=n)return;
 REAL sum=0;for(size_t p=offsets[row];p<offsets[row+1];++p)sum+=values[p]*x[columns[p]];y[row]=sum;
}
extern "C" __global__ void mv_coo(const size_t* rows,const size_t* columns,const REAL* values,const REAL* x,REAL* y,size_t n,size_t nnz){
 size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=n)return;
 size_t begin=0,end=nnz;while(begin<end){size_t mid=begin+(end-begin)/2;if(rows[mid]<row)begin=mid+1;else end=mid;}
 REAL sum=0;for(size_t p=begin;p<nnz && rows[p]==row;++p)sum+=values[p]*x[columns[p]];y[row]=sum;
}
