#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include "types.h"
#include "verification.h"
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

int main(int argc,char**argv){
 if(argc!=2){printf("Usage: %s <N>\nN: domain size in each dimension\n",argv[0]);return -1;}
 const int N=atoi(argv[1]);
 const int steps=control("HECBENCH_LBM_STEPS",2000),warmup=control("HECBENCH_LBM_WARMUP",1000),runs=control("HECBENCH_LBM_RUNS",10);
 if(N<3||N>1024||steps<=warmup||warmup<=0||runs<=0)throw std::runtime_error("invalid domain or validation controls");
 BoxCU domain;domain.nx=N;domain.ny=N;domain.nz=N;
 const size_t nl=size_t(N)*N*N;const double ulb=.02,dx=1./(N-2.),dt=dx*ulb,nu=ulb*(N-2.)/100.,omega=1./(3.*nu+.5);
 iris::Platform platform;check(platform.init(&argc,&argv,true));const int device=hip_device();
 {
 std::vector<float> velocity(3*nl,0),unused_g(3*nl,0);
 std::vector<double> rho(nl,0),f0(19*nl,0),f1(19*nl,0),mlups(runs);
 std::vector<flag_type> flags(nl,fluid),bd(nl,0);std::vector<int> values(nl,0);
 iris::DMem mv(velocity.data(),velocity.size()*sizeof(float)),mg(unused_g.data(),unused_g.size()*sizeof(float)),
 mr(rho.data(),rho.size()*sizeof(double)),m0(f0.data(),f0.size()*sizeof(double)),m1(f1.data(),f1.size()*sizeof(double)),
 mf(flags.data(),flags.size()),md(bd.data(),bd.size()),mb(values.data(),values.size()*sizeof(int)),
 mp(outer_bounds_priority,sizeof(outer_bounds_priority)),mc(const_cast<char*>(dirs),sizeof(dirs));
 outer_wall wall{bounce,bounce,bounce,bounce,moving_wall,bounce};
 size_t initg[]={size_t(N),size_t(N),size_t(N)},initl[]={size_t(N),1,1};
 auto submit=[&](const char* name,size_t* g,size_t* l,int n,void** args,int* info){iris::Task t;check(t.kernel(name,3,nullptr,g,l,n,args,info));check(t.submit(device,nullptr,true));};
 {
 void* p[]={&mf,&mb,&md,&domain,&wall,&mp};int info[]={iris_rw,iris_rw,iris_rw,sizeof(domain),sizeof(wall),iris_r};
 submit("make_flag",initg,initl,6,p,info);
 }
 {
 void* p[]={&mf,&md,&mb,&domain,&mc};int info[]={iris_rw,iris_rw,iris_rw,sizeof(domain),iris_r};
 submit("find_wall",initg,initl,5,p,info);
 }
 for(int run=0;run<runs;++run){
 printf("omega = %f\n",omega);
 {
 void* p[]={&mv,&mr,&m0,&m1,&domain};int info[]={iris_w,iris_w,iris_w,iris_w,sizeof(domain)};
 submit("init_velocity",initg,initl,5,p,info);
 }
 iris::DMem* current=&m0;iris::DMem* next=&m1;
 size_t cg[]={size_t((N-1)/64+1)*64,size_t(N),size_t(N)},cl[]={64,1,1};
 int num_bench_iter=0;auto start=std::chrono::steady_clock::now();
 printf("Starting %d warmup iterations\n",warmup);
 for(int iter=0;iter<steps;++iter){
 bool output=iter<warmup&&iter>0&&(iter%2000==0||iter==149);
 if(iter==warmup){printf("Starting %d benchmark iterations\n",steps-warmup);start=std::chrono::steady_clock::now();}
 if(iter>=warmup)++num_bench_iter;
 void* p[]={current,next,&mv,&mf,&md,&mb,&domain,arg(ulb),arg(omega),&output,&iter};
 int info[]={iris_r,iris_rw,iris_rw,iris_r,iris_r,iris_r,sizeof(domain),sizeof(ulb),sizeof(omega),sizeof(output),sizeof(iter)};
 submit("collide",cg,cl,11,p,info);std::swap(current,next);
 if(output){
 iris::Task t;check(t.flush_out(mv));check(t.submit(device,nullptr,true));
 double energy=0;
 for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){
 const size_t i=IDX(x,y,z,N,N,N);
 energy+=velocity[i]*velocity[i]+velocity[i+nl]*velocity[i+nl]+velocity[i+2*nl]*velocity[i+2*nl];
 }
 energy*=.5;
 printf("energy %f iteration %d \n",energy*dx*dx/(dt*dt),iter);
 printf("ENERGY run=%d iter=%d LU=%.17g\n",run,iter,energy);
 record_velocity(velocity.data(),nl,run,iter);
 if(iter==149&&N==102){
 const double ref=2.09868507623;const bool ok=std::isfinite(energy)&&fabs(energy-ref)<1.e-7;
 printf("Regression test at iteration %d: Average energy LU = %f: %s\n",iter,energy,ok?"OK":"FAILED");
 if(!ok)return 2;
 }
 }
 }
 const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
 mlups[run]=((double(nl)*num_bench_iter)/(elapsed*1e-9f))/1.e6;
 }
 printf("\nPerformance: MLUPS\n");for(double rate:mlups)printf("%.4f\n",rate);
 }
 check(platform.finalize());return 0;
}
