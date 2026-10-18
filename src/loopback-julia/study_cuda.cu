#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include "../loopback-cuda/loopback.h"
// Named one-block study layout; original kernel body retained byte-identically.
#undef TAUSWORTHE_NUM_BLOCKS
#define TAUSWORTHE_NUM_BLOCKS 1
#include "../loopback-cuda/kernels.cu"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(10);}}while(0)
__global__ void rng_consistency(unsigned draws,const unsigned*seeds,double*out){int i=threadIdx.x;unsigned z1=seeds[i],z2=seeds[i+256],z3=seeds[i+512],z4=seeds[i+768];float first=0,second=0;for(unsigned j=0;j<draws;j++){float u=getRandomValueTauswortheUniform(z1,z2,z3,z4);if(j==0)first=u;if(j==1)second=u;}out[6*i]=z1;out[6*i+1]=z2;out[6*i+2]=z3;out[6*i+3]=z4;out[6*i+4]=first;out[6*i+5]=second;}
template<typename T>T*up(const std::vector<T>&h){T*d;CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
std::vector<double>load(const char*f,int n){std::vector<double>x(n);FILE*q=fopen(f,"rb");if(!q||fread(x.data(),8,n,q)!=n)exit(3);fclose(q);return x;}
struct Case{unsigned T;std::vector<std::vector<float>>h;std::vector<float*>d;float*mean,*variance;double*probe;};
int main(){auto prefix=load("/home/imo/HeCBench_julia/src/loopback-julia/fixture.bin",8199),gold=load("/home/imo/HeCBench_julia/src/loopback-julia/oracle.bin",20096);if(prefix[0]!=256||prefix[1]!=4||prefix[2]!=512)return 4;std::vector<unsigned>hs(1040,0x5a5a5a5a);for(int i=0;i<1024;i++)hs[8+i]=unsigned(prefix[3+i]);auto seeds=up(hs);int pos=1027;std::vector<Case>cs;
for(int ci=0;ci<4;ci++){Case c;c.T=prefix[pos++];if(c.T>15)return 5;for(int k=0;k<7;k++){std::vector<float>h(272,-12345.f);for(int i=0;i<256;i++)h[8+i]=prefix[pos++];c.h.push_back(h);c.d.push_back(up(h));}c.mean=up(std::vector<float>(272,-12345.f));c.variance=up(std::vector<float>(272,-12345.f));c.probe=up(std::vector<double>(1536,-12345.));cs.push_back(c);}if(pos!=8199)return 6;
auto launch=[&](Case&c){tausworthe_lookback<<<1,256>>>(c.T,seeds+8,c.mean+8,c.variance+8,c.d[0]+8,c.d[1]+8,c.d[2]+8,c.d[3]+8,c.d[4]+8,c.d[5]+8,c.d[6]+8);};for(auto&c:cs){launch(c);rng_consistency<<<1,256>>>(512*2*((c.T+1)/2),seeds+8,c.probe);CK(cudaDeviceSynchronize());}
for(int trial=1;trial<=5;trial++){std::vector<double>art=prefix,paths;bool ok=true;int go=0;double seconds=0,ma=0,mr=0;for(int ci=0;ci<4;ci++){auto&c=cs[ci];std::vector<float>reset(272,-12345.f);CK(cudaMemcpy(c.mean,reset.data(),1088,cudaMemcpyHostToDevice));CK(cudaMemcpy(c.variance,reset.data(),1088,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();launch(c);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();seconds+=dt;paths.push_back(dt);
for(float*d:{c.mean,c.variance}){std::vector<float>v(272);CK(cudaMemcpy(v.data(),d,1088,cudaMemcpyDeviceToHost));for(int j=0;j<272;j++){double x=v[j],r=gold[go++],e=fabs(x-r);if(j<8||j>=264||ci>=2)ok&=x==r;else{ok&=std::isfinite(x)&&x>=0&&e<=.002+.002*fabs(r);ma=std::max(ma,e);mr=std::max(mr,e/(fabs(r)+1.e-300));}art.push_back(x);}}
rng_consistency<<<1,256>>>(512*2*((c.T+1)/2),seeds+8,c.probe);CK(cudaDeviceSynchronize());std::vector<double>pr(1536);CK(cudaMemcpy(pr.data(),c.probe,pr.size()*8,cudaMemcpyDeviceToHost));for(double x:pr){ok&=x==gold[go++];art.push_back(x);}std::vector<unsigned>ss(1040);CK(cudaMemcpy(ss.data(),seeds,4160,cudaMemcpyDeviceToHost));ok&=ss==hs;for(unsigned x:ss){ok&=double(x)==gold[go++];art.push_back(x);}for(int k=0;k<7;k++){std::vector<float>v(272);CK(cudaMemcpy(v.data(),c.d[k],1088,cudaMemcpyDeviceToHost));ok&=memcmp(v.data(),c.h[k].data(),1088)==0;for(float x:v){ok&=double(x)==gold[go++];art.push_back(double(x));}}}
if(go!=20096||art.size()!=28295)return 7;std::string file=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(file.c_str(),"wb");fwrite(art.data(),8,art.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":20096,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum4_full512path_lookbacklike_simulation_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":28295,\"byte_order\":\"little\"},\"path_times_seconds\":[%.17g,%.17g,%.17g,%.17g],\"paths_per_parameter\":512,\"RNG_probe_outside_timing\":true}\n",trial,seconds,ok?"true":"false",ma,mr,file.c_str(),paths[0],paths[1],paths[2],paths[3]);}cudaFree(seeds);for(auto&c:cs){for(auto d:c.d)cudaFree(d);cudaFree(c.mean);cudaFree(c.variance);cudaFree(c.probe);}}
