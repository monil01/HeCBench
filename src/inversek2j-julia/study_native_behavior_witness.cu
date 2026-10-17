// Diagnostic only: unchanged source CPU/GPU behavior and forward-kinematics witness.
#define main legacy_main
#include "../inversek2j-cuda/main.cu"
#undef main
#include <cstdio>
#include <vector>
int main(){float x[]={0,1.5f,2.5f,-1.5f,3,2},y[]={3,1.5f,.5f,-1.5f,0,0};constexpr int n=6;float cpu[3*n],gpu[3*n],*dx,*dy,*da;cudaMalloc(&dx,n*4);cudaMalloc(&dy,n*4);cudaMalloc(&da,3*n*4);cudaMemcpy(dx,x,n*4,cudaMemcpyHostToDevice);cudaMemcpy(dy,y,n*4,cudaMemcpyHostToDevice);invkin_cpu(x,y,cpu,n);invkin_kernel<<<1,128>>>(dx,dy,da,n);cudaDeviceSynchronize();cudaMemcpy(gpu,da,3*n*4,cudaMemcpyDeviceToHost);for(int j=0;j<n;j++){double rx=0,ry=0,ax=0,ay=0,acc=0;for(int k=0;k<3;k++){double z=gpu[3*j+k]*acos(-1.)/180;acc+=z;rx+=cos(acc);ry+=sin(acc);ax+=cos(z);ay+=sin(z);}printf("WITNESS target=(%.9g,%.9g) nativeCPU=(%.9g,%.9g,%.9g) nativeGPU=(%.9g,%.9g,%.9g) FK_relative_residual=%.17g FK_absolute_residual=%.17g finite=%d\n",x[j],y[j],cpu[3*j],cpu[3*j+1],cpu[3*j+2],gpu[3*j],gpu[3*j+1],gpu[3*j+2],hypot(rx-x[j],ry-y[j]),hypot(ax-x[j],ay-y[j]),int(std::isfinite(gpu[3*j])&&std::isfinite(gpu[3*j+1])&&std::isfinite(gpu[3*j+2])));}cudaFree(dx);cudaFree(dy);cudaFree(da);return cudaGetLastError()!=cudaSuccess;}
