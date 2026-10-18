
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#define main source_quant_main
#include "/home/imo/HeCBench_julia/src/quantAQLM-julia/native_corrected/main.cu"
#undef main
const char*BASE="/home/imo/HeCBench_julia/src/quantAQLM-julia/";const int G=16,R=20;
std::vector<uint16_t>readv(const std::string&name,size_t n){std::vector<uint16_t>v(n);std::ifstream f(std::string(BASE)+name,std::ios::binary);f.read((char*)v.data(),n*2);if(!f){fprintf(stderr,"Missingfixture%s\n",name.c_str());exit(3);}return v;}
struct Case{int m,k,blocks,threads;size_t nc,ni,nb,no;std::vector<uint16_t*>a;std::vector<size_t>lens;std::vector<uint16_t>oracle;};
void launch(Case&c,int batch){Code1x16MatVec<<<c.blocks,c.threads,4608>>>((int4*)c.a[0],(int4*)(c.a[1]+batch*c.k),(int4*)(c.a[2]+batch*c.m),(int4*)c.a[3],c.m,c.k,make_int4(c.m/3,2*c.m/3,c.m,c.m*10),65536);}
int main(){
 std::vector<Case>cases(2);int sms;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
 for(int i=0;i<2;i++){auto&c=cases[i];c.m=i?96:12288;c.k=i?512:4096;c.nc=c.m*c.k/8;c.ni=4*c.k;c.nb=3*65536*8;c.no=4*c.m;c.lens={c.nc,c.ni,c.no,c.nb};c.a.resize(4);
  int waves=0,tm;do{waves++;tm=ceildiv(c.m,waves*sms);}while(tm>THREAD_M);c.blocks=ceildiv(c.m,tm);c.threads=tm*32;
  auto input=readv("study_case"+std::to_string(i+1)+"_u16.bin",c.nc+c.ni+c.nb);c.oracle=readv("oracle_case"+std::to_string(i+1)+"_u16.bin",c.no);
  size_t pos=0;for(int j=0;j<4;j++){std::vector<uint16_t>h(c.lens[j]+G,0x7b00);if(j!=2){std::copy(input.begin()+pos,input.begin()+pos+c.lens[j],h.begin());pos+=c.lens[j];}cudaMalloc(&c.a[j],h.size()*2);cudaMemcpy(c.a[j],h.data(),h.size()*2,cudaMemcpyHostToDevice);}
  for(int b=0;b<4;b++)launch(c,b);cudaDeviceSynchronize();
 }
 for(int trial=1;trial<=5;trial++){
  std::vector<uint32_t>prefix,values,guards;std::vector<double>times;bool ok=true;int mismatches=0;
  for(int caseid=0;caseid<2;caseid++){auto&c=cases[caseid];double elapsed=0;
   for(int b=0;b<4;b++)for(int repeat=0;repeat<R;repeat++){cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();launch(c,b);cudaDeviceSynchronize();elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
   times.push_back(elapsed/R);
   for(int j=0;j<4;j++){std::vector<uint16_t>h(c.lens[j]+G);cudaMemcpy(h.data(),c.a[j],h.size()*2,cudaMemcpyDeviceToHost);for(size_t t=c.lens[j];t<h.size();t++)guards.push_back(h[t]);
    if(j==2){for(size_t t=0;t<c.no;t++){values.push_back(h[t]);bool pass=h[t]==c.oracle[t]&&std::isfinite(__half2float(*reinterpret_cast<__half*>(&h[t])));ok&=pass;mismatches+=!pass;}}
    else{for(size_t t=0;t<c.lens[j];t+=2)prefix.push_back(uint32_t(h[t])|(uint32_t(h[t+1])<<16));}
   }
   for(int x:{c.m,c.k,4,65536,c.m/3,2*c.m/3,c.m,c.k/64,R,9,8,16,0,0,0,caseid+1})prefix.push_back(x);
  }
  for(auto x:guards)ok&=x==0x7b00;auto err=cudaGetLastError();ok&=err==cudaSuccess;
  auto out=prefix;out.insert(out.end(),values.begin(),values.end());out.insert(out.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":null,\"error_metric_scope\":\"max_abs_error is halfbit mismatchflag, relativeerror uncomputed; separate denseFP64quality\",\"mismatch_half_values\":%d,\"timing_region\":\"sum2_fourvectors_meanR20_original_quant_matvec_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],ok?"true":"false",values.size()+guards.size(),mismatches?1:0,mismatches,times[0],times[1],path.c_str(),out.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto&c:cases)for(auto a:c.a)cudaFree(a);
}
