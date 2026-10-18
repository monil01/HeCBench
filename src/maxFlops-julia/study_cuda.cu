#include <cuda_runtime.h>
#include "../maxFlops-cuda/kernels.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double> rd(const char*n){FILE*f=fopen(n,"rb");if(!f)abort();fseek(f,0,SEEK_END);size_t z=ftell(f)/8;rewind(f);std::vector<double>a(z);fread(a.data(),8,z,f);fclose(f);return a;}
struct Case{int bits,N,op,L,rep;double v1,v2;std::vector<double>h;void* dev;};
template<class T>void launch(Case&c){T*d=(T*)c.dev;switch(c.op*10+c.L){case 11: Add1<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 12: Add2<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 14: Add4<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 18: Add8<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 21: Mul1<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 22: Mul2<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 24: Mul4<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 28: Mul8<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1));break;
case 31: MAdd1<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 32: MAdd2<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 34: MAdd4<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 38: MAdd8<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 41: MulMAdd1<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 42: MulMAdd2<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 44: MulMAdd4<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;
case 48: MulMAdd8<T><<<c.N/256,256>>>(d+8,c.rep,T(c.v1),T(c.v2));break;}CK(cudaGetLastError());}
void run(Case&c){if(c.bits==32)launch<float>(c);else launch<double>(c);}
template<class T>std::vector<double> reset(Case&c){std::vector<T>h(c.h.begin(),c.h.end());CK(cudaMemcpy(c.dev,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));std::vector<T>v(h.size());CK(cudaMemcpy(v.data(),c.dev,v.size()*sizeof(T),cudaMemcpyDeviceToHost));if(memcmp(h.data(),v.data(),h.size()*sizeof(T)))abort();return std::vector<double>(v.begin(),v.end());}
template<class T>std::vector<double> readout(Case&c){std::vector<T>v(c.N+16);CK(cudaMemcpy(v.data(),c.dev,v.size()*sizeof(T),cudaMemcpyDeviceToHost));return std::vector<double>(v.begin(),v.end());}
int main(){auto pre=rd(FIXTURE),gold=rd(ORACLE);size_t pos=1;std::vector<Case>cs;for(int ci=0;ci<(int)pre[0];ci++){Case c;c.bits=pre[pos++];c.N=pre[pos++];c.op=pre[pos++];c.L=pre[pos++];c.rep=pre[pos++];c.v1=pre[pos++];c.v2=pre[pos++];c.h.assign(c.N+16,-12345);for(int i=0;i<c.N;i++)c.h[i+8]=pre[pos++];CK(cudaMalloc(&c.dev,(c.N+16)*(c.bits/8)));if(c.bits==32)reset<float>(c);else reset<double>(c);cs.push_back(c);}if(pos!=pre.size())abort();for(auto&c:cs){run(c);CK(cudaDeviceSynchronize());}
for(int tr=1;tr<=5;tr++){auto art=pre;size_t go=0;bool ok=true;double sum=0,ma=0,mr=0;std::vector<double>times;
for(auto&c:cs){auto before=c.bits==32?reset<float>(c):reset<double>(c);CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();run(c);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();sum+=dt;times.push_back(dt);auto v=c.bits==32?readout<float>(c):readout<double>(c);for(int i=0;i<c.N+16;i++){double r=gold[go++],e=fabs(v[i]-r);bool exact=i<8||i>=c.N+8||c.rep==0||c.op==1||c.op==2;double at=c.bits==32?1e-4:1e-11,rt=c.bits==32?1e-5:1e-12;ok&=std::isfinite(v[i])&&(exact?v[i]==r:e<=at+rt*fabs(r));ma=fmax(ma,e);mr=fmax(mr,e/(fabs(r)+1e-300));art.push_back(v[i]);}for(double x:before){ok&=x==gold[go++];art.push_back(x);}}
if(go!=gold.size())abort();std::string f=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";FILE*o=fopen(f.c_str(),"wb");fwrite(art.data(),8,art.size(),o);fclose(o);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum192_full_source_arithmetic_network_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",tr,sum,ok?"true":"false",go,ma,mr,f.c_str(),art.size());for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("]}\n");}}
