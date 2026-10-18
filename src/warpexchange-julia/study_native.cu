#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#define main original_warpexchange_main
#include "../warpexchange-cuda/main.cu"
#undef main
#define CK(x)do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<int>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(3);size_t n=f.tellg();f.seekg(0);std::vector<int>v(n/4);f.read((char*)v.data(),n);return v;}
int main(){auto input=load("fixture_i32.bin"),expected=load("expected_i32_guards.bin");const int ns[4]={128,1152,4096,1048704};std::vector<int*>ins,outs;std::vector<int>prefix;size_t at=0;
for(int n:ns){int*a,*b;CK(cudaMalloc(&a,n*4));CK(cudaMalloc(&b,(n+16)*4));CK(cudaMemcpy(a,input.data()+at,n*4,cudaMemcpyHostToDevice));std::vector<int>init(n+16,0x13579bdf);CK(cudaMemcpy(b,init.data(),init.size()*4,cudaMemcpyHostToDevice));std::vector<int>v(n);CK(cudaMemcpy(v.data(),a,n*4,cudaMemcpyDeviceToHost));prefix.insert(prefix.end(),v.begin(),v.end());ins.push_back(a);outs.push_back(b);at+=n;}
auto run=[&](int j){k<256,1024><<<(ns[j]+255)/256,256>>>(ins[j],outs[j],ns[j]);CK(cudaGetLastError());};for(int j=0;j<4;j++)run(j);CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<int>result;double times[4];bool sourceok=true;
for(int j=0;j<4;j++){CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<100;rep++)run(j);CK(cudaDeviceSynchronize());times[j]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/100;std::vector<int>v(ns[j]+16);CK(cudaMemcpy(v.data(),outs[j],v.size()*4,cudaMemcpyDeviceToHost));if(j!=2)for(int base=0;base<ns[j];base+=128)for(int item=0;item<4;item++)for(int lane=0;lane<31;lane++)sourceok &=v[base+4*(lane+1)+item]-v[base+4*lane+item]==1;result.insert(result.end(),v.begin(),v.end());}
bool ok=result==expected&&sourceok;all &=ok;long long err=0;for(size_t i=0;i<result.size();i++)err=std::max(err,llabs((long long)result[i]-expected[i]));auto out=prefix;out.insert(out.end(),result.begin(),result.end());const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/warpexchange_trial%d.bin",dir?dir:".",trial);std::ofstream f(file,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();double time=0;for(double t:times)time+=t;
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"source_linearcase_difference_check_pass\":%s,\"checked_values\":%zu,\"max_abs_error\":%lld,\"max_rel_error\":null,\"internal_repeats\":100,\"jit_excluded\":true,\"timing_region\":\"sum_four_source_warppermutation_kernel_loopmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,time,times[0],times[1],times[2],times[3],ok?"true":"false",sourceok?"true":"false",result.size(),err,file,out.size());fflush(stdout);}
puts(all?"PASS":"FAIL");return all?0:1;}
