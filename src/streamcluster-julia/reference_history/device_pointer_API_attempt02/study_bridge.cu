// Source controller interoperation; arithmetic is retained in source_bridge.
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include "source_bridge/streamcluster.h"
using Callback=void(*)(void*,void*,void*,void*,void*,int,int,long,int);
static Callback target=nullptr;
static std::vector<uint32_t> capture;
static std::vector<float> fixture;
static bool valid=true;
static long gains=0,oracle_values=0,stale_coordinates=0;
static double max_cpu_error=0,max_oracle_budget=0;
static int work_capacity=0;
static constexpr uint32_t SENT=0x4b123456;
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA status %s\n",cudaGetErrorString(e));abort();}}
static void u(uint32_t a){capture.push_back(a);}
static void f(float a){uint32_t v;memcpy(&v,&a,4);u(v);}
static void study_launch(void*,void*,void*,void*,void*,int,int,long,int);
static void study_prepare(Points*,long,int,int,int*);
static void study_capture(Points*,long,int,float,int*,char*,float*);
static void study_final(Points*,long*);
static void study_precentroid(Points*);
#define main original_streamcluster_main
#include "source_bridge/streamcluster.cu"
#undef main
static_assert(sizeof(Point_Struct)==24 && offsetof(Point_Struct,assign)==8 && offsetof(Point_Struct,cost)==16,"GPU Point ABI");
static void study_launch(void*p,void*coord,void*work,void*centers,void*sw,int n,int dim,long x,int K){
 if(target)target(p,coord,work,centers,sw,n,dim,x,K);
 else compute_cost<<<(n+255)/256,256,dim*sizeof(float)>>>((Point_Struct*)p,(float*)coord,(float*)work,(int*)centers,(char*)sw,n,dim,x,K);
 check(cudaGetLastError());
}
static void study_prepare(Points*points,long x,int K,int kmax,int*table){
 const int n=points->num,d=points->dim;
 if(c==0)work_capacity=(kmax+1)*n;
 std::vector<uint32_t>w(work_capacity+16,SENT);check(cudaMemcpy(work_mem_d,w.data(),w.size()*4,cudaMemcpyHostToDevice));
 std::vector<uint32_t>g(16,SENT);check(cudaMemcpy(coord_d+n*d,g.data(),64,cudaMemcpyHostToDevice));check(cudaMemcpy(center_table_d+n,g.data(),64,cudaMemcpyHostToDevice));
 std::vector<unsigned char>s(16,0x5b);check(cudaMemcpy(switch_membership_d+n,s.data(),16,cudaMemcpyHostToDevice));
 std::vector<unsigned char>pg(16*sizeof(Point_Struct),0x5b);check(cudaMemcpy(p_d+n,pg.data(),pg.size(),cudaMemcpyHostToDevice));
 // Actual device inputs, defined struct fields only: C++ padding is not read by the kernel.
 std::vector<float>coords(n*d);std::vector<Point_Struct>p(n);std::vector<int>ct(n);
 check(cudaMemcpy(coords.data(),coord_d,coords.size()*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(p.data(),p_d,n*sizeof(Point_Struct),cudaMemcpyDeviceToHost));check(cudaMemcpy(ct.data(),center_table_d,n*4,cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++){valid &= p[i].weight==points->p[i].weight&&p[i].assign==points->p[i].assign&&p[i].cost==points->p[i].cost&&ct[i]==table[i];for(int j=0;j<d;j++)if(coords[j*n+i]!=points->p[i].coord[j])stale_coordinates++;}
 u(0x11111111);u(n);u(d);u(x);u(K);
 for(float a:coords)f(a);
 for(int i=0;i<n;i++){f(p[i].weight);u(p[i].assign);f(p[i].cost);u(ct[i]);}
 check(cudaDeviceSynchronize());
}
static void study_capture(Points*points,long x,int K,float z,int*table,char*sw,float*work){
 const int n=points->num,d=points->dim;u(0x22222222);u(n);u(K);u(x);f(z);gains++;
 // Separate double-distance oracle, and Float32/FMA source-policy oracle.
 for(int i=0;i<n;i++){
  float sc=0;double dc=0;
  for(int j=0;j<d;j++){float a=coord_h[j*n+i]-coord_h[j*n+x];sc=std::fma(a,a,sc);double b=double(coord_h[j*n+i])-coord_h[j*n+x];dc+=b*b;}
  sc*=points->p[i].weight;dc*=points->p[i].weight;
  double eb=2e-6+2e-6*std::abs(dc);max_cpu_error=std::max(max_cpu_error,std::abs(double(sc)-dc));max_oracle_budget=std::max(max_oracle_budget,std::abs(double(sc)-dc)/eb);valid &= std::isfinite(sc)&&std::abs(double(sc)-dc)<=eb;
  const float cur=points->p[i].cost;const bool switches=sc<cur;valid &= (sw[i]==(switches?'1':0));u((unsigned char)sw[i]);
  int slot=switches?K:table[points->p[i].assign];float want=switches?sc-cur:cur-sc;
  for(int j=0;j<=K;j++){float actual=work[i*(K+1)+j],expect=j==slot?want:0;valid &= std::isfinite(actual)&&actual==expect;f(actual);oracle_values++;}
 }
 std::vector<uint32_t>w(work_capacity+16);check(cudaMemcpy(w.data(),work_mem_d,w.size()*4,cudaMemcpyDeviceToHost));for(int i=n*(K+1);i<work_capacity+16;i++){valid &= w[i]==SENT;u(w[i]);}
 std::vector<uint32_t>g(16);for(void*p:{(void*)(coord_d+n*d),(void*)(center_table_d+n)}){check(cudaMemcpy(g.data(),p,64,cudaMemcpyDeviceToHost));for(auto a:g){valid &=a==SENT;u(a);}}
 std::vector<unsigned char>b(16+16*sizeof(Point_Struct));check(cudaMemcpy(b.data(),switch_membership_d+n,16,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data()+16,p_d+n,16*sizeof(Point_Struct),cudaMemcpyDeviceToHost));for(auto a:b){valid &=a==0x5b;u(a);}
}
static void study_precentroid(Points*p){
 u(0x33333333);u(p->num);u(p->dim);
 for(int i=0;i<p->num;i++){
  long a=p->p[i].assign;valid &=a>=0&&a<p->num;u(a);f(p->p[i].weight);f(p->p[i].cost);for(int j=0;j<p->dim;j++)f(p->p[i].coord[j]);
  double ds=0;for(int j=0;j<p->dim;j++){double t=double(p->p[i].coord[j])-p->p[a].coord[j];ds+=t*t;}ds*=p->p[i].weight;
  valid &=std::isfinite(p->p[i].cost)&&std::abs(double(p->p[i].cost)-ds)<=2e-5+2e-5*std::abs(ds);
 }
}
static void study_final(Points*p,long*ids){
 u(0x44444444);u(p->num);u(p->dim);std::vector<bool>median(p->num,false);for(int i=0;i<p->num;i++){valid &=p->p[i].assign>=0&&p->p[i].assign<p->num;median[p->p[i].assign]=true;}
 double mass=0;std::vector<double>moment(p->dim,0);
 for(int i=0;i<p->num;i++){u(ids[i]);u(p->p[i].assign);u(median[i]);f(p->p[i].weight);f(p->p[i].cost);for(int j=0;j<p->dim;j++){f(p->p[i].coord[j]);valid &=std::isfinite(p->p[i].coord[j]);if(median[i])moment[j]+=double(p->p[i].weight)*p->p[i].coord[j];}if(median[i])mass+=p->p[i].weight;}
 valid &=mass==512;for(int j=0;j<p->dim;j++){double want=0;for(int i=0;i<512;i++)want+=fixture[i*p->dim+j];valid &=std::abs(moment[j]-want)<=1e-3;}
}
class FixtureStream:public PStream{bool done=false;public:size_t read(float*dest,int dim,int num){if(done)return 0;std::copy(fixture.begin(),fixture.end(),dest);done=true;return 512;}int ferror(){return 0;}int feof(){return done;}};
extern "C" void study_run(Callback cb,int trial,const char*fixture_path,const char*outdir){
 target=cb;fixture.resize(512*8);FILE*fi=fopen(fixture_path,"rb");if(!fi||fread(fixture.data(),4,fixture.size(),fi)!=fixture.size())abort();fclose(fi);
 capture.clear();for(float a:fixture)f(a);for(uint32_t a:{512,8,2,4,512,64,1,256,16,1})u(a);
 c=0;gains=oracle_values=stale_coordinates=0;max_cpu_error=max_oracle_budget=0;valid=true;nproc=1;srand48(SEED);kernel_time=serial=cpu_gpu_memcpy=memcpy_back=gpu_malloc=0;
 FixtureStream stream;std::string out=std::string(outdir)+"/centers.txt";streamCluster(&stream,2,4,8,512,64,out.data());check(cudaDeviceSynchronize());valid &=stale_coordinates==0&&gains>0;
 free(coord_h);free(gl_lower);free(work_mem_h);free(p_h);check(cudaFree(work_mem_d));check(cudaFree(coord_d));check(cudaFree(center_table_d));check(cudaFree(switch_membership_d));check(cudaFree(p_d));
 if(trial){std::string path=std::string(outdir)+"/trial"+std::to_string(trial)+".bin";FILE*fo=fopen(path.c_str(),"wb");fwrite(capture.data(),4,capture.size(),fo);fclose(fo);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_budget_error\":%.17g,\"timing_region\":\"sum_pgain_source_memset_compute_cost_devicecomplete_CPUcontrollerinterop\",\"jit_excluded\":true,\"internal_repeats\":1,\"gain_calls\":%ld,\"oracle_work_values\":%ld,\"stale_coordinate_values\":%ld,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,kernel_time*1e-9,valid?"true":"false",capture.size()-4106,max_cpu_error,max_oracle_budget,gains,oracle_values,stale_coordinates,path.c_str(),capture.size());fflush(stdout);if(!valid)exit(1);}
 else if(!valid){fprintf(stderr,"Warmup validation failed gains%ld stale%ld budget%.9g\n",gains,stale_coordinates,max_oracle_budget);exit(1);}
}
#ifdef STUDY_MAIN
int main(){const char*dir=getenv("STUDY_OUTPUT_DIR");if(!dir)dir=".";study_run(nullptr,0,"input_f32.bin",dir);for(int i=1;i<=5;i++)study_run(nullptr,i,"input_f32.bin",dir);return 0;}
#endif
