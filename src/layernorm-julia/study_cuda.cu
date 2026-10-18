#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#define main original_layernorm_main
#include "../layernorm-cuda/main.cu"
#undef main
struct Case{int N,C,mode;std::vector<std::vector<float>>h,reset;std::vector<float*>d,o;};
std::vector<double>load(const char*f,int n){std::vector<double>x(n);FILE*q=fopen(f,"rb");if(!q||fread(x.data(),8,n,q)!=n)exit(3);fclose(q);return x;}
float*up(const std::vector<float>&h){float*d;cudaCheck(cudaMalloc(&d,h.size()*4));cudaCheck(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
int main(){auto prefix=load("/home/imo/HeCBench_julia/src/layernorm-julia/fixture.bin",18505),gold=load("/home/imo/HeCBench_julia/src/layernorm-julia/oracle.bin",566136);int pos=1;std::vector<Case>cs;
for(int ci=0;ci<int(prefix[0]);ci++){Case c;c.N=prefix[pos++];c.C=prefix[pos++];c.mode=prefix[pos++];for(int n:{c.N*c.C,c.C,c.C}){std::vector<float>v(n+16,-12345.f);for(int i=0;i<n;i++)v[8+i]=prefix[pos++];c.h.push_back(v);c.d.push_back(up(v));}for(int n:{c.N*c.C,c.N,c.N}){std::vector<float>v(n+16,-12345.f);c.reset.push_back(v);c.o.push_back(up(v));}cs.push_back(c);}if(pos!=18505)return 4;
auto launch=[](Case&c,int k,int bt){layernorm_forward(k,c.o[0]+8,c.o[1]+8,c.o[2]+8,c.d[0]+8,c.d[1]+8,c.d[2]+8,1,c.N,c.C,bt);};for(auto&c:cs)for(int k=0;k<3;k++)for(int bt:{32,64,128,256,512,1024})launch(c,k,bt);
for(int trial=1;trial<=5;trial++){std::vector<double>art=prefix,paths;bool ok=true;int go=0;double seconds=0,ma=0,mr=0;
for(auto&c:cs)for(int k=0;k<3;k++)for(int bt:{32,64,128,256,512,1024}){for(int j=0;j<3;j++)cudaCheck(cudaMemcpy(c.o[j],c.reset[j].data(),c.reset[j].size()*4,cudaMemcpyHostToDevice));cudaCheck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int rep=0;rep<10;rep++)launch(c,k,bt);double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/10.;seconds+=dt;paths.push_back(dt);
for(int j=0;j<3;j++){std::vector<float>v(c.reset[j].size());cudaCheck(cudaMemcpy(v.data(),c.o[j],v.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<int(v.size());i++){double x=v[i],r=gold[go++],e=fabs(x-r);if(i<8||i>=int(v.size())-8)ok&=x==r;else{ok&=std::isfinite(x)&&e<=1e-5+2e-6*fabs(r);ma=std::max(ma,e);mr=std::max(mr,e/(fabs(r)+1e-300));if(j==2)ok&=x>0;if(j==0&&((i-8)%c.C)%19==0)ok&=v[i]==c.h[2][8+(i-8)%c.C];if(j==0&&c.mode==1)ok&=v[i]==c.h[2][8+(i-8)%c.C];}art.push_back(x);}}
for(int j=0;j<3;j++){std::vector<float>v(c.h[j].size());cudaCheck(cudaMemcpy(v.data(),c.d[j],v.size()*4,cudaMemcpyDeviceToHost));ok&=memcmp(v.data(),c.h[j].data(),v.size()*4)==0;for(float x:v){ok&=double(x)==gold[go++];art.push_back(double(x));}}}
if(go!=566136||art.size()!=584641)return 5;std::string file=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(file.c_str(),"wb");fwrite(art.data(),8,art.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":566136,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum108_layernorm_3alg6width_kernel_sync_batches_per10\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":584641,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,seconds,ok?"true":"false",ma,mr,file.c_str());for(int i=0;i<108;i++)printf("%s%.17g",i?",":"",paths[i]);printf("]}\n");}for(auto&c:cs){for(auto d:c.d)cudaFree(d);for(auto d:c.o)cudaFree(d);}}
