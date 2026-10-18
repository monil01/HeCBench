#define main legacy_main
#include "../ludb-cuda/main.cu"
#undef main
#include <vector>
#include <tuple>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <cmath>
#include "study_fixture.h"
int main(){const int batch=4,n=48;float *a;float **ap;int *pv,*info;cudaMalloc(&a,9216*4);cudaMalloc(&ap,batch*sizeof(float*));cudaMalloc(&pv,n*batch*4);cudaMalloc(&info,batch*4);float* hp[4];for(int b=0;b<4;b++)hp[b]=a+b*n*n;cudaMemcpy(ap,hp,sizeof(hp),cudaMemcpyHostToDevice);cublasHandle_t h;cublasCreate(&h);int ver;cublasGetVersion(h,&ver);std::cout<<"STUDY_LIBRARY version="<<ver<<std::endl;
auto run=[&](std::vector<double>&out){cudaMemcpy(a,INPUT_A,sizeof(INPUT_A),cudaMemcpyHostToDevice);cudaMemset(pv,0,192*4);cudaMemset(info,0,16);cudaDeviceSynchronize();auto s=std::chrono::steady_clock::now();auto status=getrfBatched(h,n,ap,n,pv,info,batch);cudaDeviceSynchronize();double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count();std::vector<float> lu(9216);std::vector<int> p(192),inf(4);cudaMemcpy(lu.data(),a,9216*4,cudaMemcpyDeviceToHost);cudaMemcpy(p.data(),pv,192*4,cudaMemcpyDeviceToHost);cudaMemcpy(inf.data(),info,16,cudaMemcpyDeviceToHost);out.assign(INPUT_A,INPUT_A+9216);for(int x:{48,4,1,4,1})out.push_back(x);for(float x:lu)out.push_back(x);for(int x:p)out.push_back(x);for(int x:inf)out.push_back(x);out.push_back(status);bool ok=status==0&&inf[0]==0&&inf[1]==0&&inf[2]==1&&inf[3]==0;double err=0;for(int b=0;b<4;b++){int perm[48];for(int i=0;i<48;i++)perm[i]=i;for(int k=0;k<48;k++){int q=p[b*48+k]-1;if(q<k||q>=48){ok=false;continue;}std::swap(perm[k],perm[q]);}for(int j=0;j<48;j++)for(int i=0;i<48;i++){double x=0;for(int k=0;k<48;k++){double l=i==k?1:(i>k?lu[b*2304+k*48+i]:0);double u=k<=j?lu[b*2304+j*48+k]:0;x+=l*u;}err=std::max(err,std::abs(x-INPUT_A[b*2304+j*48+perm[i]]));}}ok=ok&&err<2e-6;return std::make_tuple(t,ok,err);};
std::vector<double> out;run(out);for(int trial=1;trial<=5;trial++){auto[t,ok,e]=run(out);std::string file="trial_"+std::to_string(trial)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<std::setprecision(17)<<t<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<e<<",\"timing_region\":\"one_cublas_getrf_batched_call\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\""<<file<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}cublasDestroy(h);}
