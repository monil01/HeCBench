
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#define main source_segsort_main
#include "/home/imo/HeCBench_julia/src/segsort-cuda/main.cu"
#undef main
const char*BASE="/home/imo/HeCBench_julia/src/segsort-julia/";const int G=16,IS=0x5a5a5a5a;const uint64_t VS=0x12345678fedcba98ULL;
template<class T>std::vector<T>readv(const std::string&name,size_t n){std::vector<T>a(n);std::ifstream f(std::string(BASE)+name,std::ios::binary);f.read((char*)a.data(),n*sizeof(T));if(!f){fprintf(stderr,"Fixture%s\n",name.c_str());exit(3);}return a;}
struct Case{bool pairs;int n,ns;int *A,*B,*offsets,*ids,*dc,*hc;uint64_t*V,*W;cudaStream_t stream;cudaEvent_t event;std::vector<int>keys,off,oracle,counts;std::vector<uint64_t>val,oraclev;};
std::vector<uint32_t>wordsi(const std::vector<int>&a){std::vector<uint32_t>x(a.size());memcpy(x.data(),a.data(),a.size()*4);return x;}
void appendv(std::vector<uint32_t>&x,const std::vector<uint64_t>&a,size_t start,size_t end){for(size_t i=start;i<end;i++){x.push_back(uint32_t(a[i]));x.push_back(uint32_t(a[i]>>32));}}
void reset(Case&c){cudaMemcpyAsync(c.A,c.keys.data(),c.n*4,cudaMemcpyHostToDevice,c.stream);if(c.pairs)cudaMemcpyAsync(c.V,c.val.data(),c.n*8,cudaMemcpyHostToDevice,c.stream);cudaStreamSynchronize(c.stream);}
double execute(Case&c){
 cudaStreamSynchronize(c.stream);auto t=std::chrono::steady_clock::now();
 if(c.pairs)bb_segsort_run(c.A,c.V,c.B,c.W,c.offsets,c.offsets+1,c.ns,c.ids,c.hc,c.dc,c.stream,c.event);
 else bb_segsort_run(c.A,c.B,c.offsets,c.offsets+1,c.ns,c.ids,c.hc,c.dc,c.stream,c.event);
 auto err=cudaStreamSynchronize(c.stream);if(err!=cudaSuccess){fprintf(stderr,"CUDA%s\n",cudaGetErrorString(err));exit(1);}
 return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
int main(){
 std::ifstream meta(std::string(BASE)+"study_sizes.txt");int n,ns;meta>>n>>ns;if(!meta)return 3;auto off=readv<int>("offsets_i32.bin",ns+1);
 std::vector<Case>cases(2);
 for(int mode=0;mode<2;mode++){auto&c=cases[mode];c.pairs=mode;c.n=n;c.ns=ns;c.off=off;c.keys=readv<int>("keys_case"+std::to_string(mode+1)+"_i32.bin",n);c.oracle=readv<int>("oracle_case"+std::to_string(mode+1)+"_i32.bin",n);c.counts.assign(13,0);
  for(int i=0;i<ns;i++){int sz=off[i+1]-off[i],b=0;while(b<12&&sz>(1<<b))b++;c.counts[b]++;}
  cudaStreamCreate(&c.stream);cudaEventCreate(&c.event);cudaMallocHost(&c.hc,(14+G)*4);for(int i=0;i<14+G;i++)c.hc[i]=IS;
  int**a[]={&c.A,&c.B,&c.offsets,&c.ids,&c.dc};size_t lens[]={size_t(n),size_t(n),size_t(ns+1),size_t(ns),14};
  for(int j=0;j<5;j++){std::vector<int>v(lens[j]+G,IS);cudaMalloc(a[j],v.size()*4);cudaMemcpy(*a[j],v.data(),v.size()*4,cudaMemcpyHostToDevice);}
  cudaMemcpy(c.offsets,off.data(),(ns+1)*4,cudaMemcpyHostToDevice);
  if(mode){c.val=readv<uint64_t>("values_case2_u64.bin",n);c.oraclev=readv<uint64_t>("oracle_values_case2_u64.bin",n);std::vector<uint64_t>v(n+G,VS);cudaMalloc(&c.V,v.size()*8);cudaMalloc(&c.W,v.size()*8);cudaMemcpy(c.V,v.data(),v.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(c.W,v.data(),v.size()*8,cudaMemcpyHostToDevice);}
  reset(c);execute(c); // all32entrymethods/mergepath configurations warm outside5trials.
 }
 for(int trial=1;trial<=5;trial++){
  std::vector<uint32_t>prefix,values,guards;std::vector<std::vector<int>>rawids;std::vector<double>times;bool ok=true;size_t mismatches=0;int idchecks=0;
  for(int mode=0;mode<2;mode++){auto&c=cases[mode];reset(c);
   std::vector<int>before(n),offs(ns+1);cudaMemcpy(before.data(),c.A,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(offs.data(),c.offsets,(ns+1)*4,cudaMemcpyDeviceToHost);auto w=wordsi(before);prefix.insert(prefix.end(),w.begin(),w.end());
   if(mode){std::vector<uint64_t>bv(n);cudaMemcpy(bv.data(),c.V,n*8,cudaMemcpyDeviceToHost);appendv(prefix,bv,0,n);}
   auto ow=wordsi(offs);prefix.insert(prefix.end(),ow.begin(),ow.end());for(int x:{n,ns,mode,13,2048,5000,1,0})prefix.push_back(x);
   times.push_back(execute(c));std::vector<int>out(n),dc(14),ids(ns);cudaMemcpy(out.data(),c.B,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(dc.data(),c.dc,14*4,cudaMemcpyDeviceToHost);cudaMemcpy(ids.data(),c.ids,ns*4,cudaMemcpyDeviceToHost);
   for(int i=0;i<n;i++){bool pass=out[i]==c.oracle[i];ok&=pass;mismatches+=!pass;}
   auto outw=wordsi(out);values.insert(values.end(),outw.begin(),outw.end());
   if(mode){std::vector<uint64_t>v(n);cudaMemcpy(v.data(),c.W,n*8,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++){bool pass=v[i]==c.oraclev[i];ok&=pass;mismatches+=!pass;}appendv(values,v,0,n);}
   int sum=0;std::vector<int>seen(ns,0);for(int b=0;b<13;b++){ok&=c.hc[b]==sum;for(int pos=sum;pos<sum+c.counts[b];pos++){int id=ids[pos];bool pass=id>=0&&id<ns;if(pass){seen[id]++;int sz=off[id+1]-off[id],bin=0;while(bin<12&&sz>(1<<bin))bin++;pass=bin==b;}ok&=pass;idchecks++;}sum+=c.counts[b];ok&=dc[b]==sum;}
   for(auto v:seen)ok&=v==1;ok&=dc[13]==5000&&c.hc[13]==ns;
   for(int i=0;i<14;i++){values.push_back(uint32_t(dc[i]));values.push_back(uint32_t(c.hc[i]));}
   int*arr[]={c.A,c.B,c.offsets,c.ids,c.dc};size_t lens[]={size_t(n),size_t(n),size_t(ns+1),size_t(ns),14};
   for(int j=0;j<5;j++){int g[G];cudaMemcpy(g,arr[j]+lens[j],G*4,cudaMemcpyDeviceToHost);for(int z:g){guards.push_back(uint32_t(z));ok&=z==IS;}}
   for(int i=14;i<14+G;i++){guards.push_back(uint32_t(c.hc[i]));ok&=c.hc[i]==IS;}
   if(mode)for(auto a:{c.V,c.W}){std::vector<uint64_t>g(G);cudaMemcpy(g.data(),a+n,G*8,cudaMemcpyDeviceToHost);for(auto z:g)ok&=z==VS;appendv(guards,g,0,G);}
   rawids.push_back(ids);
  }
  auto all=prefix;all.insert(all.end(),values.begin(),values.end());all.insert(all.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)all.data(),all.size()*4);f.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":null,\"mismatch_output_values\":%zu,\"error_metric_scope\":\"Computedexactkey/value/countermismatchflag; relativeerror notapplicable\",\"timing_region\":\"sum_keys_and_pairs_full_bb_bin_dispatch_network_large_merge_stream_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"internal_bin_ids_checked\":%d,\"internal_bin_ids_raw_not_public_output\":[",trial,times[0]+times[1],ok?"true":"false",values.size()+guards.size(),mismatches?1:0,mismatches,idchecks);
  for(int mode=0;mode<2;mode++){if(mode)printf(",");printf("[");for(size_t i=0;i<rawids[mode].size();i++)printf("%s%d",i?",":"",rawids[mode][i]);printf("]");}
  printf("],\"component_time_seconds\":[%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",times[0],times[1],path.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto&c:cases){for(auto a:{c.A,c.B,c.offsets,c.ids,c.dc})cudaFree(a);if(c.pairs){cudaFree(c.V);cudaFree(c.W);}cudaFreeHost(c.hc);cudaEventDestroy(c.event);cudaStreamDestroy(c.stream);}
}
