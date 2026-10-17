#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#define main original_bscan_main
#include "../bscan-cuda/main.cu"
#undef main
int main(){std::vector<int>ns={32,64,128,256,512,1024},expected,got(2016);std::vector<int*>ins,outs;for(int n:ns){std::vector<int>a(n);int count=0;for(int i=0;i<n;i++){a[i]=(i*37+11)%101-50;expected.push_back(count);count+=a[i]>0;}int *di,*out;cudaMalloc(&di,n*4);cudaMalloc(&out,n*4);cudaMemcpy(di,a.data(),n*4,cudaMemcpyHostToDevice);ins.push_back(di);outs.push_back(out);}auto launch=[&](){for(int k=0;k<6;k++)binary_scan<<<1,ns[k]>>>(outs[k],ins[k]);};launch();cudaDeviceSynchronize();bool all=true;int r=10;for(int trial=1;trial<=5;trial++){auto start=std::chrono::steady_clock::now();for(int j=0;j<r;j++)launch();auto error=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/r;int offset=0;for(int k=0;k<6;k++){cudaMemcpy(got.data()+offset,outs[k],ns[k]*4,cudaMemcpyDeviceToHost);offset+=ns[k];}bool ok=error==cudaSuccess&&got==expected;all &=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/output_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)got.data(),got.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",got.size(),r,path.c_str(),got.size());}puts(all?"PASS":"FAIL");return all?0:1;}
