#define main original_main
#include "../vmc-cuda/vmc2.cu"
#undef main
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
static void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(x));exit(2);}}
std::vector<double> readall(const char*f){std::ifstream in(f,std::ios::binary|std::ios::ate);if(!in)exit(3);size_t n=in.tellg();if(n%8)exit(3);in.seekg(0);std::vector<double>v(n/8);in.read((char*)v.data(),n);if(!in)exit(3);return v;}
template<class T>struct Guard{T*base;T*d;int n;Guard(int n):n(n){ck(cudaMalloc(&base,(n+128)*sizeof(T)));d=base+64;std::vector<T>h(n+128,0);std::fill(h.begin(),h.begin()+64,16384);std::fill(h.end()-64,h.end(),16384);ck(cudaMemcpy(base,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}void capture(std::vector<double>&v){std::vector<T>h(n+128);ck(cudaMemcpy(h.data(),base,h.size()*sizeof(T),cudaMemcpyDeviceToHost));for(T x:h)v.push_back(x);}~Guard(){cudaFree(base);}};
int main(){auto pre=readall(FIXTURE),gold=readall(ORACLE);std::ifstream f(KINDS,std::ios::binary);std::vector<unsigned char>kind(gold.size());f.read((char*)kind.data(),kind.size());if(!f||pre.size()!=9)return 3;
 Guard<float>x1(Npoint),y1(Npoint),z1(Npoint),x2(Npoint),y2(Npoint),z2(Npoint),psi(Npoint),stats(4*Npoint),sum(4),blocks(NBLOCK);Guard<unsigned>states(Npoint);
 auto init=[&](unsigned seed){initran<<<NBLOCK,256>>>(seed,states.d);initialize<<<NBLOCK,256>>>(x1.d,y1.d,z1.d,x2.d,y2.d,z2.d,psi.d,states.d);zero_stats<<<NBLOCK,256>>>(Npoint,stats.d);};
 auto prop=[&](int steps){propagate<<<NBLOCK,256>>>(Npoint,steps,x1.d,y1.d,z1.d,x2.d,y2.d,z2.d,psi.d,stats.d,states.d);};
 auto capture=[&](std::vector<double>&v,bool full){for(auto a:{&x1,&y1,&z1,&x2,&y2,&z2,&psi})a->capture(v);if(full)stats.capture(v);states.capture(v);};
 init(5551212);prop(1);zero_stats<<<NBLOCK,256>>>(Npoint,stats.d);prop(1);sum_stats(Npoint,stats.d,sum.d,blocks.d);ck(cudaDeviceSynchronize());
 for(int trial=1;trial<=5;trial++){auto art=pre;std::vector<double>times;for(int ci=0;ci<2;ci++){int burn=ci?5:2,steps=ci?7:3;init(5551212+ci);ck(cudaDeviceSynchronize());capture(art,false);prop(burn);ck(cudaDeviceSynchronize());capture(art,true);
 for(int sample=0;sample<2;sample++){ck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();zero_stats<<<NBLOCK,256>>>(Npoint,stats.d);prop(steps);sum_stats(Npoint,stats.d,sum.d,blocks.d);ck(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());capture(art,true);sum.capture(art);blocks.capture(art);}}
 if(art.size()!=pre.size()+gold.size())return 5;bool ok=true;double ma=0,mp=0;for(size_t i=0;i<gold.size();i++){double v=art[pre.size()+i],e=std::abs(v-gold[i]),atol=kind[i]==1?1e-12:2e-5,rtol=kind[i]==1?2e-5:3e-5;ok&=std::isfinite(v)&&(kind[i]==2?v==gold[i]:e<=atol+rtol*std::abs(gold[i]));ma=std::max(ma,e);if(kind[i]==1)mp=std::max(mp,e);}double total=0;for(double x:times)total+=x;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)art.data(),art.size()*8);out.close();if(!out)return 6;
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_source_sample_zero_propagate_and_eight_tree_reductions\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"sample_times_seconds\":[%.17g,%.17g,%.17g,%.17g],\"wavefunction_max_abs_error\":%.17g,\"exact_RNG_accept_counts_and_guards_checked\":true,\"authored_device_kernels\":5}\n",trial,total,ok?"true":"false",art.size(),ma,path.c_str(),art.size(),times[0],times[1],times[2],times[3],mp);}}
