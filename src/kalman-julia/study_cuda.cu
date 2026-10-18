#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#define main original_kalman_main
#include "../kalman-cuda/main.cu"
#undef main
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(10);}}while(0)
struct Case{int ns,no,nf;bool inter,conf;std::vector<std::vector<double>>h,reset;std::vector<double*>d,o;};
std::vector<double>load(const char*f,int n){std::vector<double>x(n);FILE*q=fopen(f,"rb");if(!q||fread(x.data(),8,n,q)!=n)exit(3);fclose(q);return x;}
double*up(const std::vector<double>&h){double*d;CK(cudaMalloc(&d,h.size()*8));CK(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice));return d;}
int main(){auto prefix=load("/home/imo/HeCBench_julia/src/kalman-julia/fixture.bin",59311),gold=load("/home/imo/HeCBench_julia/src/kalman-julia/oracle.bin",506124);int pos=1;std::vector<Case>cs;
for(int ci=0;ci<int(prefix[0]);ci++){Case c;c.ns=prefix[pos++];c.no=prefix[pos++];c.nf=prefix[pos++];c.inter=prefix[pos++];c.conf=prefix[pos++];int lens[]={c.ns*c.no,c.ns*64,c.ns*8,c.ns*64,c.ns*64,c.ns*8,c.ns};for(int n:lens){std::vector<double>v(n+16,-12345.);for(int i=0;i<n;i++)v[8+i]=prefix[pos++];c.h.push_back(v);c.d.push_back(up(v));}for(int n:{c.ns*c.no,c.ns*c.no,c.ns,c.ns*c.nf,c.ns*c.nf}){std::vector<double>v(n+16,-12345.);c.reset.push_back(v);c.o.push_back(up(v));}cs.push_back(c);}if(pos!=59311)return 4;
auto launch=[](Case&c,int nd){kalman<8><<<(c.ns+255)/256,256>>>(c.d[0]+8,c.no,c.d[1]+8,c.d[2]+8,c.d[3]+8,c.d[4]+8,c.d[5]+8,c.inter,c.d[6]+8,c.ns,c.o[0]+8,c.o[1]+8,c.o[2]+8,nd,c.nf,c.o[3]+8,c.conf,c.o[4]+8);};for(int ci=0;ci<4;ci++)for(int nd=0;nd<8;nd++){if(ci==3&&nd!=0&&nd!=7)continue;launch(cs[ci],nd);CK(cudaDeviceSynchronize());}
for(int trial=1;trial<=5;trial++){std::vector<double>art=prefix,paths;bool ok=true;int go=0;double seconds=0,ma=0,mr=0;
for(int ci=0;ci<4;ci++)for(int nd=0;nd<8;nd++){if(ci==3&&nd!=0&&nd!=7)continue;auto&c=cs[ci];for(int k=0;k<5;k++)CK(cudaMemcpy(c.o[k],c.reset[k].data(),c.reset[k].size()*8,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();launch(c,nd);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();seconds+=dt;paths.push_back(dt);
for(int k=0;k<5;k++){std::vector<double>v(c.reset[k].size());CK(cudaMemcpy(v.data(),c.o[k],v.size()*8,cudaMemcpyDeviceToHost));for(int i=0;i<int(v.size());i++){double x=v[i],r=gold[go++],e=fabs(x-r);bool exact=i<8||i>=int(v.size())-8||(k==4&&!c.conf)||(k==2&&nd>=c.no);if(exact)ok&=x==r;else{ok&=std::isfinite(x)&&e<=1e-9+1e-9*fabs(r);ma=std::max(ma,e);mr=std::max(mr,e/(fabs(r)+1e-300));if(k==1||k==4)ok&=x>0.;}art.push_back(x);}}
for(int k=0;k<7;k++){std::vector<double>v(c.h[k].size());CK(cudaMemcpy(v.data(),c.d[k],v.size()*8,cudaMemcpyDeviceToHost));ok&=memcmp(v.data(),c.h[k].data(),v.size()*8)==0;for(double x:v){ok&=x==gold[go++];art.push_back(x);}}}
if(go!=506124||art.size()!=565435)return 5;std::string file=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(file.c_str(),"wb");fwrite(art.data(),8,art.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":506124,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum26_full_Kalman_filter_forecast_kernels\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":565435,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,seconds,ok?"true":"false",ma,mr,file.c_str());for(int i=0;i<26;i++)printf("%s%.17g",i?",":"",paths[i]);printf("]}\n");}for(auto&c:cs){for(auto d:c.d)cudaFree(d);for(auto d:c.o)cudaFree(d);}}
