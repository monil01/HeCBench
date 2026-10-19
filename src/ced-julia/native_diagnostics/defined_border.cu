#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "../source_kernels.inc"
int main(){const int rows=66,cols=98,n=rows*cols;std::vector<unsigned char> input(n);for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)input[r*cols+c]=(r*13+c*7+(r/9)*(c/7)*11)%256;unsigned char *a,*b,*theta;cudaMalloc(&a,n);cudaMalloc(&b,n);cudaMalloc(&theta,n);cudaMemcpy(a,input.data(),n,cudaMemcpyHostToDevice);cudaMemset(b,0,n);cudaMemset(theta,0,n);dim3 threads(16,16),blocks((cols-2)/16,(rows-2)/16);gaussian_kernel<<<blocks,threads,18*18*4>>>(a,b,rows,cols);sobel_kernel<<<blocks,threads,18*18*4>>>(b,a,theta,rows,cols);non_max_supp_kernel<<<blocks,threads,18*18*4>>>(a,b,theta,rows,cols);hyst_kernel<<<blocks,threads,18*18*4>>>(b,a,rows,cols);auto s=cudaDeviceSynchronize();printf("Defined-zero-border full4stage diagnostic CUDAstatus=%d; sourcekernelsunchanged\n",(int)s);cudaFree(a);cudaFree(b);cudaFree(theta);return s==cudaSuccess?0:1;}
