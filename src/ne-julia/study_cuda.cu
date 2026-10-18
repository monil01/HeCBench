#include <cuda_runtime.h>
#define main legacy_main
#include "../ne-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
static_assert(sizeof(float3)==12&&sizeof(float4)==16,"layout");
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double>rd(const char*n){FILE*f=fopen(n,"rb");fseek(f,0,SEEK_END);size_t z=ftell(f)/8;rewind(f);std::vector<double>a(z);if(fread(a.data(),8,z,f)!=z)abort();fclose(f);return a;}
struct Case{int W,H,N;std::vector<float3>h;float3*d;float4*out;};
void launch(Case&c){ne<<<(c.N+255)/256,256>>>(c.d+64,c.out+16,c.W,c.H,c.N);CK(cudaGetLastError());}
int main(){auto pre=rd(FIXTURE),gold=rd(ORACLE);size_t pos=1;std::vector<Case>cs;for(int ci=0;ci<5;ci++){Case c;c.W=pre[pos++];c.H=pre[pos++];c.N=c.W*c.H;c.h.assign(c.N+72,make_float3(-12345,-12345,-12345));for(int j=0;j<c.N;j++){uint32_t bits[3];for(int k=0;k<3;k++)bits[k]=pre[pos++];memcpy(&c.h[j+64],bits,12);}CK(cudaMalloc(&c.d,c.h.size()*12));CK(cudaMalloc(&c.out,(c.N+24)*16));CK(cudaMemcpy(c.d,c.h.data(),c.h.size()*12,cudaMemcpyHostToDevice));cs.push_back(c);}if(pos!=pre.size())abort();for(auto&c:cs){launch(c);CK(cudaDeviceSynchronize());}
for(int tr=1;tr<=5;tr++){auto art=pre;size_t go=0;bool ok=true;double ma=0,mr=0,total=0;std::vector<double>times;for(auto&c:cs){std::vector<float4>out(c.N+24,make_float4(-12345,-12345,-12345,-12345));CK(cudaMemcpy(c.out,out.data(),out.size()*16,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();launch(c);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();times.push_back(dt);total+=dt;CK(cudaMemcpy(out.data(),c.out,out.size()*16,cudaMemcpyDeviceToHost));for(int i=0;i<c.N+24;i++){float xyz[4];memcpy(xyz,&out[i],16);bool control=i<16||i>=c.N+16;bool zeroref=!control&&gold[go]==0&&gold[go+1]==0&&gold[go+2]==0;for(int k=0;k<4;k++){double r=gold[go++],x=xyz[k],e=fabs(x-r);bool exact=control||k==3||zeroref;ok&=std::isfinite(x)&&(exact?x==r:e<=1e-6+3e-6*fabs(r));ma=fmax(ma,e);mr=fmax(mr,e/(fabs(r)+1e-300));art.push_back(x);}if(!control&&!zeroref){auto q=c.h[i-16+64];double len=xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2],dot=q.x*xyz[0]+q.y*xyz[1]+q.z*xyz[2];ok&=fabs(len-1)<=1e-5&&dot<=1e-5;}}
std::vector<float3>q(c.h.size());CK(cudaMemcpy(q.data(),c.d,q.size()*12,cudaMemcpyDeviceToHost));ok&=memcmp(q.data(),c.h.data(),q.size()*12)==0;for(auto v:q){uint32_t b[3];memcpy(b,&v,12);for(auto x:b){ok&=x==gold[go++];art.push_back(x);}}}
if(go!=gold.size())abort();std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(art.data(),8,art.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum5_complete_pointcloud_normal_estimation_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",tr,total,ok?"true":"false",go,ma,mr,path.c_str(),art.size());for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("]}\n");}}
