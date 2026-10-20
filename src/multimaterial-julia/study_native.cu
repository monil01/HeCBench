#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "corrected_compact.cu"
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* load(FILE*f,int n){std::vector<double>a(n);if(fread(a.data(),8,n,f)!=(size_t)n)exit(3);std::vector<T>b(a.begin(),a.end());T*d;CHECK(cudaMalloc(&d,n*sizeof(T)));CHECK(cudaMemcpy(d,b.data(),n*sizeof(T),cudaMemcpyHostToDevice));return d;}
double* output(int n){std::vector<double>a(n+17,0);for(int i=n;i<n+17;i++)a[i]=-987654.25;double*d;CHECK(cudaMalloc(&d,a.size()*8));CHECK(cudaMemcpy(d,a.data(),a.size()*8,cudaMemcpyHostToDevice));return d;}
template<class T>void append_input(std::vector<double>&a,T*d,int n){std::vector<T>b(n);CHECK(cudaMemcpy(b.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));for(auto v:b)a.push_back((double)v);}
void append_output(std::vector<double>&a,double*d,int n){size_t o=a.size();a.resize(o+n);CHECK(cudaMemcpy(a.data()+o,d,n*8,cudaMemcpyDeviceToHost));}
int main(int argc,char**argv){if(argc!=3)return 3;FILE*f=fopen(argv[1],"rb");if(!f)return 3;
 auto imaterial=load<int>(f,7680);
 auto nextfrac=load<int>(f,17280);
 auto matids=load<int>(f,17280);
 auto mmc_index=load<int>(f,5761);
 auto mmc_i=load<int>(f,5760);
 auto mmc_j=load<int>(f,5760);
 auto rho_cell=load<double>(f,7680);
 auto rho_list=load<double>(f,17280);
 auto vf_list=load<double>(f,17280);
 auto t_cell=load<double>(f,7680);
 auto t_list=load<double>(f,17280);
 auto V=load<double>(f,7680);
 auto x=load<double>(f,7680);
 auto y=load<double>(f,7680);
 auto n=load<double>(f,50);
fclose(f);
 auto rho_ave=output(7680);
 auto p_cell=output(7680);
 auto p_list=output(17280);
 auto neighbor_cell=output(7680);
 auto neighbor_list=output(17280);
constexpr int sx=96,sy=80,nc=7680,mc=5760,ml=17280;
dim3 threads(32,4),blocks(3,20);
auto one=[&](){ccc_loop1<<<blocks,threads>>>(imaterial,nextfrac,rho_cell,rho_list,vf_list,V,rho_ave,sx,sy,mmc_index);
#ifndef FUSED
ccc_loop1_2<<<(mc+127)/128,128>>>(rho_list,vf_list,V,rho_ave,mmc_index,mc,mmc_i,mmc_j,sx,sy);
#endif
CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());};
auto two=[&](){ccc_loop2<<<blocks,threads>>>(imaterial,matids,nextfrac,rho_cell,rho_list,t_cell,t_list,vf_list,n,p_cell,p_list,sx,sy,mmc_index);
#ifndef FUSED
ccc_loop2_2<<<(ml+127)/128,128>>>(matids,rho_list,t_list,vf_list,n,p_list,mmc_index,ml);
#endif
CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());};
auto three=[&](){ccc_loop3<<<blocks,threads>>>(imaterial,nextfrac,matids,rho_cell,rho_list,neighbor_cell,neighbor_list,x,y,sx,sy,mmc_index);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());};
one();two();three();
std::vector<double> expected(57685);f=fopen(argv[2],"rb");if(!f)return 3;if(fread(expected.data(),8,expected.size(),f)!=expected.size())return 3;fclose(f);
for(int trial=1;trial<=5;trial++){
CHECK(cudaDeviceSynchronize());auto t0=std::chrono::steady_clock::now();one();auto t1=std::chrono::steady_clock::now();two();auto t2=std::chrono::steady_clock::now();three();auto t3=std::chrono::steady_clock::now();
std::vector<double>capture;
 append_input(capture,imaterial,7680);
 append_input(capture,nextfrac,17280);
 append_input(capture,matids,17280);
 append_input(capture,mmc_index,5761);
 append_input(capture,mmc_i,5760);
 append_input(capture,mmc_j,5760);
 append_input(capture,rho_cell,7680);
 append_input(capture,rho_list,17280);
 append_input(capture,vf_list,17280);
 append_input(capture,t_cell,7680);
 append_input(capture,t_list,17280);
 append_input(capture,V,7680);
 append_input(capture,x,7680);
 append_input(capture,y,7680);
 append_input(capture,n,50);
 append_output(capture,rho_ave,7680+17);
 append_output(capture,p_cell,7680+17);
 append_output(capture,p_list,17280+17);
 append_output(capture,neighbor_cell,7680+17);
 append_output(capture,neighbor_list,17280+17);
bool good=true;double ma=0,mr=0;for(size_t q=0;q<expected.size();q++){double v=capture[149811+q],e=expected[q],er=fabs(v-e);if(!std::isfinite(v)||er>1e-8+2e-13*fabs(e))good=false;ma=fmax(ma,er);if(e!=0)mr=fmax(mr,er/fabs(e));}
std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";f=fopen(path.c_str(),"wb");if(!f)return 3;fwrite(capture.data(),8,capture.size(),f);fclose(f);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"algorithm_seconds\":[%.17g,%.17g,%.17g],\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_valid\":true,\"jit_excluded\":true,\"region\":\"compact_alg1_then_alg2_then_alg3_each_device_completion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu}}\n",trial,std::chrono::duration<double>(t3-t0).count(),std::chrono::duration<double>(t1-t0).count(),std::chrono::duration<double>(t2-t1).count(),std::chrono::duration<double>(t3-t2).count(),good?"true":"false",expected.size(),ma,mr,path.c_str(),capture.size());fflush(stdout);if(!good)return 1;
}CHECK(cudaDeviceSynchronize());return 0;}
