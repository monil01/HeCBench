#include <hip/hip_runtime.h>
extern "C" __global__ void spmm_coo(const int* rows,const int* columns,const float* values,
                                   const float* b,float* c,int m,int n,int nnz) {
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(m)*n) return;
    int row=index/n,col=index%n;
    int begin=0,end=nnz;
    while(begin<end) {int mid=begin+(end-begin)/2;if(rows[mid]<row)begin=mid+1;else end=mid;}
    float sum=0.0f;
    for(int p=begin;p<nnz && rows[p]==row;++p)sum+=values[p]*b[columns[p]*n+col];
    c[index]=sum;
}
extern "C" __global__ void spmm_csr(const int* offsets,const int* columns,const float* values,
                                   const float* b,float* c,int m,int n,int nnz) {
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(m)*n) return;
    int row=index/n,col=index%n;float sum=0.0f;
    for(int p=offsets[row];p<offsets[row+1];++p)sum+=values[p]*b[columns[p]*n+col];
    c[index]=sum;
}
