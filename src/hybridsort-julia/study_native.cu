
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include "/home/imo/HeCBench_julia/src/hybridsort-cuda/bucketsort.cu"
#include "/home/imo/HeCBench_julia/src/hybridsort-cuda/mergesort.cu"
const int N=16384,CAP=N+DIVISIONS*4,G=32;
const char *BASE="/home/imo/HeCBench_julia/src/hybridsort-julia/";
void ck(){auto e=cudaGetLastError();if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
std::vector<float> readf(std::string name){std::vector<float>v(N);std::ifstream f(std::string(BASE)+name,std::ios::binary);f.read((char*)v.data(),N*4);if(!f)exit(3);return v;}
int main(){
 std::vector<std::vector<float>> inputs,oracles;
 for(int c=1;c<=2;c++){inputs.push_back(readf("study_case"+std::to_string(c)+"_f32.bin"));oracles.push_back(readf("oracle_case"+std::to_string(c)+"_f32.bin"));}
 auto run=[&](int caseid,int trial,std::vector<double>&prefix,std::vector<double>&out,std::vector<double>&guards,double&elapsed){
  std::vector<float>in(CAP+G,0),result(CAP+G,0);std::copy(inputs[caseid].begin(),inputs[caseid].end(),in.begin());
  for(int i=CAP;i<CAP+G;i++){in[i]=12345.5f;result[i]=12345.5f;}
  std::vector<int>sizes(DIVISIONS),nulls(DIVISIONS);std::vector<unsigned int>orig(DIVISIONS+1);
  float lo=*std::min_element(in.begin(),in.begin()+N),hi=*std::max_element(in.begin(),in.begin()+N);
  cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();
  bucketSort(in.data(),result.data(),N,sizes.data(),nulls.data(),lo,hi,orig.data());
  int newsize=0;for(auto s:sizes)newsize+=s*4;
  runMergeSort(newsize,DIVISIONS,(float4*)result.data(),(float4*)in.data(),sizes.data(),nulls.data(),orig.data());
  cudaDeviceSynchronize();elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();ck();
  for(int i=0;i<N;i++){unsigned int bits;memcpy(&bits,&in[i],4);prefix.push_back(bits);}
  for(int a:{N,DIVISIONS,96,64,32,128,256,208,caseid+1})prefix.push_back(a);
  for(int i=0;i<N;i++)out.push_back(result[i]);
  for(auto a:sizes)out.push_back(a);for(auto a:nulls)out.push_back(a);for(auto a:orig)out.push_back(a);
  for(int i=CAP;i<CAP+G;i++){guards.push_back(in[i]);guards.push_back(result[i]);}
  bool ok=true;for(int i=0;i<N;i++)if(result[i]!=oracles[caseid][i])ok=false;
  unsigned int total=0;for(int i=0;i<DIVISIONS;i++){if(nulls[i]<0||nulls[i]>3||sizes[i]*4-nulls[i]!=(int)(orig[i+1]-orig[i]))ok=false;total+=sizes[i]*4-nulls[i];}
  if(total!=N)ok=false;for(auto g:guards)if(g!=12345.5)ok=false;return ok;
 };
 for(int c=0;c<2;c++){std::vector<double>p,o,g;double e;run(c,0,p,o,g,e);}
 for(int trial=1;trial<=5;trial++){
  std::vector<double>prefix,out,guards;double total=0;bool ok=true;
  for(int c=0;c<2;c++){double e;ok=run(c,trial,prefix,out,guards,e)&&ok;total+=e;}
  std::vector<double>all=prefix;all.insert(all.end(),out.begin(),out.end());all.insert(all.end(),guards.begin(),guards.end());
  std::string f=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream os(f,std::ios::binary);os.write((char*)all.data(),all.size()*8);os.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":39042,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum2_full_histogram_cpu_pivot_bucket_merge_pack_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",f.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }
}
