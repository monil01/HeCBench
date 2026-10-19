#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <string>
#include "kernels.cu"
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s:%d\n",cudaGetErrorString(e),__LINE__);exit(7);}}while(0)
int main(int argc,char**argv){if(argc!=3)return 1;std::ifstream f(argv[1]);int n;f>>n;if(n!=1048576)return 2;std::vector<float>input(9*n+32,91.25f),output(21*n+32,91.25f);for(int i=0;i<9*n;i++)if(!(f>>input[i+16]))return 3;float extra;if(f>>extra)return 4;float *di,*out;C(cudaMalloc(&di,input.size()*4));C(cudaMalloc(&out,output.size()*4));C(cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice));C(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));C(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();svd3_SOA<<<n/256+1,256>>>(di+16,out+16,n);C(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();C(cudaGetLastError());C(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));std::vector<float>echo(input.size());C(cudaMemcpy(echo.data(),di,echo.size()*4,cudaMemcpyDeviceToHost));if(echo!=input)return 5;for(int i=0;i<16;i++)if(output[i]!=91.25f||output[21*n+16+i]!=91.25f)return 6;std::ofstream a(std::string(argv[2])+"/actual_input_f32.bin",std::ios::binary),b(std::string(argv[2])+"/actual_output_f32.bin",std::ios::binary);a.write((char*)echo.data(),echo.size()*4);b.write((char*)output.data(),output.size()*4);printf("SVD_FULL_DATASET_PROBE n=%d all_input_readbacks_exact=1 all_guards_exact=1 seconds=%.17g\n",n,seconds);return 0;}
