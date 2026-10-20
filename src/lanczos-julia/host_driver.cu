#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <streambuf>
#include <iostream>
#include "study_api.h"
#include "configured_native/lanczos.cu"
#undef cudaMalloc
#undef cudaFree
#include "graph_controls.h"
#include "independent_eigen_gold.h"
struct Allocation{unsigned char*base;unsigned char*p;size_t bytes;int id;};
static std::vector<Allocation> allocations;
static LCcallback callback=nullptr;static FILE* output=nullptr;static size_t fields=0,checked=0;static int stages=0,current_case=0,allocation_id=0;static double excluded=0,max_abs=0,max_ratio=0,max_spectral=0;
using Clock=std::chrono::steady_clock;static auto now(){return Clock::now();}static double since(Clock::time_point t){return std::chrono::duration<double>(now()-t).count();}
static void fail(const char*s){fprintf(stderr,"Lanczos study failure case%d stage%d: %s\n",current_case,stages,s);fflush(stderr);exit(2);}static void ck(cudaError_t x){if(x!=cudaSuccess)fail(cudaGetErrorString(x));}
static void put(double v){if(!std::isfinite(v))fail("nonfinite artifact");if(output)fwrite(&v,8,1,output);fields++;}static void vec(const double*x,size_t n){for(size_t i=0;i<n;i++)put(x[i]);}
static void ints(const std::vector<int>&a){for(int x:a)put(x);}static void bits(const double*x,size_t n){for(size_t i=0;i<n;i++){uint64_t u;memcpy(&u,x+i,8);put(uint32_t(u));put(uint32_t(u>>32));}}
static void check(double got,long double gold,double at=1e-10,double rt=1e-10){if(!std::isfinite(got)||!std::isfinite(gold))fail("nonfinite scientific comparison");double e=double(fabsl((long double)got-gold));max_abs=std::max(max_abs,e);double b=at+rt*double(fabsl(gold));max_ratio=std::max(max_ratio,e/b);checked++;if(e>b){fprintf(stderr,"actual%.17g gold%.21Lg error%.17g budget%.17g\n",got,gold,e,b);fail("full independent stage oracle");}}
static Allocation* allocation(void*p){for(auto&a:allocations)if(a.p==p)return &a;fail("unknown GPUallocation");return nullptr;}
static void guards(void*p,bool serialize=false){auto&a=*allocation(p);unsigned char b[512];ck(cudaMemcpy(b,a.base,256,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b+256,a.p+a.bytes,256,cudaMemcpyDeviceToHost));for(auto x:b)if(x!=0xa5)fail("allocation guard");if(serialize){put(200);put(a.id);put(a.bytes);for(auto x:b)put(x);}}
cudaError_t study_malloc_raw(void**p,size_t bytes){Allocation a{};a.bytes=bytes;a.id=allocation_id++;ck(cudaMalloc((void**)&a.base,bytes+512));a.p=a.base+256;ck(cudaMemset(a.base,0xa5,256));ck(cudaMemset(a.p+bytes,0xa5,256));allocations.push_back(a);*p=a.p;return cudaSuccess;}
cudaError_t study_free(void*p){auto t=now();guards(p,true);excluded+=since(t);auto a=*allocation(p);auto it=std::find_if(allocations.begin(),allocations.end(),[&](const Allocation&v){return v.p==p;});allocations.erase(it);return cudaFree(a.base);}
static std::vector<double> readf(void*p,size_t n){if(allocation(p)->bytes!=8*n)fail("Float64 allocation/span mismatch");std::vector<double>a(n);ck(cudaMemcpy(a.data(),p,n*8,cudaMemcpyDeviceToHost));return a;}
static std::vector<int> readi(void*p,size_t n){if(allocation(p)->bytes!=4*n)fail("Int32 allocation/span mismatch");std::vector<int>a(n);ck(cudaMemcpy(a.data(),p,n*4,cudaMemcpyDeviceToHost));return a;}
extern "C" int study_call(const LCStage*s){
 if(s->n!=257||s->stage<0||s->stage>3)fail("descriptor scope");ck(cudaDeviceSynchronize());stages++;auto t=now();
 std::vector<double>x,y,v;std::vector<int>rp,ci;size_t outn=s->n;void*outp=nullptr;
 put(100+s->stage);put(s->n);put(s->nnz);put(s->group);put(s->blocks);put(s->scalar);
 if(s->stage==0){x=readf(s->p[0],s->n);y=readf(s->p[1],s->n);vec(x.data(),x.size());vec(y.data(),y.size());outn=s->blocks;outp=s->p[2];}
 else if(s->stage==1){x=readf(s->p[0],s->n);vec(x.data(),x.size());outp=s->p[0];}
 else if(s->stage==2){y=readf(s->p[0],s->n);x=readf(s->p[1],s->n);vec(y.data(),y.size());vec(x.data(),x.size());outp=s->p[0];}
 else{rp=readi(s->p[0],s->n+1);ci=readi(s->p[1],s->nnz);v=readf(s->p[2],s->nnz);x=readf(s->p[3],s->n);outp=s->p[4];ints(rp);ints(ci);vec(v.data(),v.size());vec(x.data(),x.size());
  auto expected=csr_matrix<double>(control_graph(current_case));if(memcmp(rp.data(),expected.row_ptr_data(),rp.size()*4)||memcmp(ci.data(),expected.col_ind_data(),ci.size()*4)||memcmp(v.data(),expected.values_data(),v.size()*8))fail("actual GPUCSR bits differ from source fixture");
  long double norm=0;for(double q:x)norm+=(long double)q*q;if(fabsl(norm-1)>1e-10)fail("normalized Lanczos inputvector");
 }
 excluded+=since(t);
 if(callback){if(callback(s))fail("Julia GPUcallback failed");}
 else if(s->stage==0)dot_product_kernel<double><<<s->blocks,256>>>(s->n,(double*)s->p[0],(double*)s->p[1],(double*)s->p[2]);
 else if(s->stage==1)multiply_inplace_kernel<double><<<s->blocks,256>>>(s->n,(double*)s->p[0],s->scalar);
 else if(s->stage==2)saxpy_inplace_kernel<double><<<s->blocks,256>>>(s->n,(double*)s->p[0],(double*)s->p[1],s->scalar);
 else warp_multiply_kernel<double><<<s->blocks,256>>>(s->group,s->n,0,(int*)s->p[0],(int*)s->p[1],(double*)s->p[2],(double*)s->p[3],(double*)s->p[4]);
 ck(cudaGetLastError());ck(cudaDeviceSynchronize()); // Included: never subtract scientific GPU wait as observation.
 t=now();auto got=readf(outp,outn);guards(outp);
 if(s->stage==0){for(int b=0;b<s->blocks;b++){long double z=0;for(int i=b*256;i<std::min(s->n,(b+1)*256);i++)z+=(long double)x[i]*y[i];check(got[b],z);}}
 else if(s->stage==1){for(int i=0;i<s->n;i++)check(got[i],(long double)x[i]*s->scalar);}
 else if(s->stage==2){for(int i=0;i<s->n;i++)check(got[i],(long double)y[i]+(long double)s->scalar*x[i]);}
 else for(int r=0;r<s->n;r++){long double z=0;for(int j=rp[r];j<rp[r+1];j++){if(ci[j]<0||ci[j]>=s->n)fail("CSRcolumn bound");z+=(long double)v[j]*x[ci[j]];}check(got[r],z);}
 vec(got.data(),got.size());excluded+=since(t);return 0;
}
extern "C" void study_tridiag(const double*a,const double*b,int n){auto t=now();if(n!=71)fail("source stepcount");put(300);put(n);vec(a,n);vec(b,n-1);excluded+=since(t);}
struct Sink:std::streambuf{int overflow(int c){return c;}};static Sink sink;
extern "C" int lanczos_run(int trial,LCcallback cb,const char*path){
 callback=cb;const char*record=path?path:"warmup_diagnostic.bin";output=fopen(record,"wb");if(!output)fail("output file");fields=checked=stages=allocation_id=0;excluded=max_abs=max_ratio=max_spectral=0;if(!allocations.empty())fail("leaked allocation");std::cout.rdbuf(&sink);
 for(int c=0;c<5;c++){auto m=csr_matrix<double>(control_graph(c));int avg=m.nonzeros()/m.row_size();int g=avg>16?32:avg>8?16:avg>4?8:avg>2?4:2;put(257);put(m.nonzeros());put(g);put(3);put(71);for(int i=0;i<258;i++)put(m.row_ptr(i));for(int i=0;i<m.nonzeros();i++)put(m.col_ind(i));bits(m.values_data(),m.nonzeros());std::vector<double> v(257,0);v[0]=1;bits(v.data(),v.size());}
 size_t prefix=fields;double total=0;
 for(int c=0;c<5;c++){current_case=c;auto m=csr_matrix<double>(control_graph(c));ck(cudaDeviceSynchronize());double before=excluded;auto t=now();auto eig=gpu_lanczos_eigen(m,3,71);ck(cudaDeviceSynchronize());total+=since(t)-(excluded-before);if(eig.size()!=3)fail("source filtered top3 count");put(400);put(c);put(eig.size());
  for(int i=0;i<3;i++){double e=fabs(eig[i]-independent_eigen_gold[c][i]);max_spectral=std::max(max_spectral,e);check(eig[i],independent_eigen_gold[c][i],1e-4,1e-6);put(eig[i]);}
  if(!allocations.empty())fail("source GPUallocation leak");
 }
 if(stages!=2125)fail("complete source stagecount");if(output)fclose(output);output=nullptr;
 if(trial)printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_oracle_error\":%.17g,\"max_abs_top3_spectral_error\":%.17g,\"authored_device_kernel_count\":4,\"scientific_kernel_calls\":%d,\"excluded_observation_seconds\":%.17g,\"timing_region\":\"full_Lanczos71_CSRrecurrence_and_TQLRAT_filter_allocation_with_common_perstage_completion_minus_observation\",\"jit_excluded\":true,\"internal_repeats\":1,\"input_prefix_count\":%zu,\"output_artifact\":{\"path\":\"output_trial%d.bin\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,checked,max_abs,max_ratio,max_spectral,stages,excluded,prefix,trial,fields);fflush(stdout);return 0;
}
#ifndef LC_LIBRARY
int main(){lanczos_run(0,nullptr,nullptr);for(int i=1;i<=5;i++){char f[64];sprintf(f,"output_trial%d.bin",i);lanczos_run(i,nullptr,f);}}
#endif
