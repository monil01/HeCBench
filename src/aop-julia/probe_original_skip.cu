#define main original_main
#include "../aop-cuda/main.cu"
#undef main
int main(){double *p,*s;int*f;CHECK_CUDA(cudaMalloc(&p,256*2*sizeof(double)));CHECK_CUDA(cudaMalloc(&s,32*sizeof(double)));CHECK_CUDA(cudaMalloc(&f,2*sizeof(int)));double h[512];for(auto &x:h)x=20;CHECK_CUDA(cudaMemcpy(p,h,sizeof(h),cudaMemcpyHostToDevice));CHECK_CUDA(cudaMemset(f,0,8));puts("ORIGINAL_PREPARE all256pathsoutofthemoney launch");fflush(stdout);prepare_svd_kernel<256><<<1,256>>>(256,4,PayoffPut(4),p,f,s);CHECK_CUDA(cudaDeviceSynchronize());int flag=-1;CHECK_CUDA(cudaMemcpy(&flag,f,4,cudaMemcpyDeviceToHost));printf("COMPLETED flag=%d\n",flag);}
