#define main source_snake_main
#include "../snake-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <cstring>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<uint32_t> read32(const char*name,size_t n){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=n*4)throw std::runtime_error(name);f.seekg(0);std::vector<uint32_t>x(n);if(!f.read((char*)x.data(),n*4))throw std::runtime_error(name);return x;}
int main(){constexpr int N=1025,G=16,R=20;auto a=read32("study_reads_u32.bin",8*N),b=read32("study_refs_u32.bin",8*N),oracle=read32("oracle_results_i32.bin",26*N);std::vector<uint32_t>ha=a,hb=b;ha.resize(8*N+G,0x5a5a5a5a);hb.resize(8*N+G,0x5a5a5a5a);uint32_t*A,*B;int*O;CK(cudaMalloc(&A,ha.size()*4));CK(cudaMalloc(&B,hb.size()*4));CK(cudaMalloc(&O,(N+G)*4));CK(cudaMemcpy(A,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(B,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));
 CK(cudaMemset(O,0x5a,(N+G)*4));for(int t=0;t<26;t++)sneaky_snake<<<(N+255)/256,256>>>(A,B,O,N,t);CK(cudaDeviceSynchronize());
 for(int trial=1;trial<=5;trial++){
  std::vector<uint32_t>prefix(16*N),values,guards;CK(cudaMemcpy(prefix.data(),A,8*N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(prefix.data()+8*N,B,8*N*4,cudaMemcpyDeviceToHost));for(auto x:{N,100,25,26,R,256,8,0})prefix.push_back(x);bool ok=true;int mismatches=0;std::vector<double>times;
  for(int t=0;t<26;t++){
   CK(cudaMemset(O,0x5a,(N+G)*4));CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<R;rep++)sneaky_snake<<<(N+255)/256,256>>>(A,B,O,N,t);CK(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;times.push_back(seconds);
   std::vector<uint32_t>out(N+G);CK(cudaMemcpy(out.data(),O,out.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<N;i++){mismatches+=out[i]!=oracle[t*N+i];values.push_back(out[i]);ok&=out[i]<=1&&out[i]==oracle[t*N+i];}for(int i=N;i<N+G;i++){guards.push_back(out[i]);ok&=out[i]==0x5a5a5a5a;}
  }
  uint32_t tail[G];for(auto ptr:{A+8*N,B+8*N}){CK(cudaMemcpy(tail,ptr,G*4,cudaMemcpyDeviceToHost));for(auto x:tail){guards.push_back(x);ok&=x==0x5a5a5a5a;}}
  auto all=prefix;all.insert(all.end(),values.begin(),values.end());all.insert(all.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)all.data(),all.size()*4);f.close();double total=0;for(auto t:times)total+=t;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":null,\"error_metric_scope\":\"Exactbinaryfilteroutputsandguardmismatchflag; relativeerrornotapplicable\",\"mismatch_output_values\":%d,\"timing_region\":\"sum26thresholds_meanR20_source_filter_kernel_launch_device_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[",trial,total,ok?"true":"false",values.size()+guards.size(),mismatches>0?1:0,mismatches);for(int t=0;t<26;t++)printf("%s%.17g",t?",":"",times[t]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }CK(cudaFree(A));CK(cudaFree(B));CK(cudaFree(O));return 0;}
