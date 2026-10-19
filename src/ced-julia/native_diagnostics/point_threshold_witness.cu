#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../source_kernels.inc"
int main(int argc,char**argv){const int n=18*18;std::vector<unsigned char>x(n,0),y(n,0);x[5*18+5]=100;x[5*18+6]=30;x[11*18+11]=50;unsigned char *a,*b;cudaMalloc(&a,n);cudaMalloc(&b,n);cudaMemcpy(a,x.data(),n,cudaMemcpyHostToDevice);cudaMemset(b,0,n);hyst_kernel<<<1,dim3(16,16),0>>>(a,b,18,18);auto e=cudaDeviceSynchronize();cudaMemcpy(y.data(),b,n,cudaMemcpyDeviceToHost);FILE*f=fopen(argv[1],"wb");fwrite(y.data(),1,n,f);fclose(f);printf("source_point_threshold strong=%u connected_weak=%u isolated_mid=%u CUDAstatus=%d\n",y[5*18+5],y[5*18+6],y[11*18+11],(int)e);cudaFree(a);cudaFree(b);return e==cudaSuccess?0:1;}
