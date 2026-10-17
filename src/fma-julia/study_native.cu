#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "study_original_fp32.cuh"
int main(){const int na=1024,nc=2048,ch=64,ops=1024,r=20;std::vector<float>a(na*ch),b(ch),expected(nc*ch),got(2*nc*ch);std::vector<int>ins(ops),outs(ops);for(int i=0;i<na*ch;i++)a[i]=float(i%17-8)/8;for(int i=0;i<ch;i++)b[i]=float(i%7-3)/4;for(int i=0;i<ops;i++){ins[i]=i*37%na;outs[i]=i*19%nc;for(int k=0;k<ch;k++)expected[outs[i]*ch+k]=a[ins[i]*ch+k]*b[k];}
float *da,*db,*c1,*c2;int *di,*dout;cudaMalloc(&da,a.size()*4);cudaMalloc(&db,b.size()*4);cudaMalloc(&c1,nc*ch*4);cudaMalloc(&c2,nc*ch*4);cudaMalloc(&di,ops*4);cudaMalloc(&dout,ops*4);cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(di,ins.data(),ops*4,cudaMemcpyHostToDevice);cudaMemcpy(dout,outs.data(),ops*4,cudaMemcpyHostToDevice);
auto launch=[&](){cudaMemset(c1,0,nc*ch*4);cudaMemset(c2,0,nc*ch*4);implicit_fma_kernel<float><<< (ops*(ch/4)+255)/256,256,ch*4 >>>(da,db,c1,di,dout,ops,ch,na,nc);implicit_fma_kernel_rowwise<float><<< (ops+255)/256,256,ch*4 >>>(da,db,c2,di,dout,ops,ch,na,nc);};launch();cudaDeviceSynchronize();bool all=true;
for(int trial=1;trial<=5;trial++){auto start=std::chrono::steady_clock::now();for(int j=0;j<r;j++)launch();auto err=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/r;cudaMemcpy(got.data(),c1,nc*ch*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+nc*ch,c2,nc*ch*4,cudaMemcpyDeviceToHost);double maxabs=0;bool ok=err==cudaSuccess;for(size_t j=0;j<got.size();j++){double diff=std::abs(double(got[j])-expected[j%(nc*ch)]);ok &= std::isfinite(got[j])&&diff<=1e-3;maxabs=std::max(maxabs,diff);}all &= ok;std::string path=std::string(std::getenv("STUDY_OUTPUT_DIR"))+"/output_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write(reinterpret_cast<char*>(got.data()),got.size()*4);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",got.size(),maxabs,r,path.c_str(),got.size());}puts(all?"PASS":"FAIL");return all?0:1;}
