#include <hip/hip_runtime.h>
extern "C" __global__ void product(const int* ao,const int* ac,const float* av,
 const int* bo,const int* bc,const float* bv,float* scratch,int* counts,int m,int n) {
 int row=blockIdx.x*blockDim.x+threadIdx.x; if(row>=m)return;
 float* out=scratch+size_t(row)*n;
 for(int j=0;j<n;++j)out[j]=0;
 for(int p=ao[row];p<ao[row+1];++p)
   for(int q=bo[ac[p]];q<bo[ac[p]+1];++q)out[bc[q]]+=av[p]*bv[q];
 int count=0;for(int j=0;j<n;++j)count+=out[j]!=0;counts[row]=count;
}
extern "C" __global__ void prefix(const int* counts,int* offsets,int m) {
 if(blockIdx.x || threadIdx.x)return;
 offsets[0]=0;for(int i=0;i<m;++i)offsets[i+1]=offsets[i]+counts[i];
}
extern "C" __global__ void emit(const float* scratch,const int* offsets,int* columns,float* values,int m,int n) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=m)return;
 int p=offsets[row];for(int j=0;j<n;++j)if(scratch[size_t(row)*n+j]!=0){
 columns[p]=j;values[p++]=scratch[size_t(row)*n+j];}
}
