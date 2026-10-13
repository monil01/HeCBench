#pragma once
#include <iris/iris.hpp>
#include <vector>
#include <utility>
static std::vector<std::pair<iris::DMem*, void*>> storage;
static iris::DMem* create_mem(size_t n) {
 void *h=calloc(1,n); auto *m=new iris::DMem(h,n); storage.push_back({m,h}); return m;
}
static void release_mem() { for(auto v:storage) { delete v.first; free(v.second); } }
static void prove_backend() {
 int n=0; iris_device_count(&n); bool found=false;
 for(int i=0;i<n;++i) { int b=0,t=0; char name[256]={}; size_t s;
 iris_device_info(i,iris_backend,&b,&s); iris_device_info(i,iris_type,&t,&s); iris_device_info(i,iris_name,name,&s);
 printf("IRIS backend%d/type%d %s\n",b,t,name); if(b==3 && t==256) found=true; }
 if(!found) exit(2);
}
static void run(const char *name,int dims,size_t *g,size_t *l,int n,void **p,int *info) {
 iris::Task task; task.kernel(name,dims,nullptr,g,l,n,p,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
static void launch_setup(iris::DMem *a,iris::DMem *b,int N,uint64_t seed,uint64_t seq) {
 size_t g[]={size_t((N+255)/256)*256},l[]={256}; void *p[]={a,b,&N,&seed,&seq};
 int info[]={iris_w,iris_w,sizeof(N),sizeof(seed),sizeof(seq)}; run("kernel_gpupcg_setup",1,g,l,5,p,info);
}
static void launch_random(iris::DMem *s,int N,iris::DMem *a,iris::DMem *b) {
 size_t g[]={size_t((N+255)/256)*256},l[]={256}; void *p[]={s,&N,a,b};
 int info[]={iris_w,sizeof(N),iris_rw,iris_rw}; run("kernel_reset_random_gpupcg",1,g,l,4,p,info);
}
static void launch_reset_i(iris::DMem *s,int N,int value) {
 size_t g[]={size_t((N+255)/256)*256},l[]={256}; void *p[]={s,&N,&value};
 int info[]={iris_w,sizeof(N),sizeof(value)}; run("kernel_reset_i",1,g,l,3,p,info);
}
static void launch_reset_f(iris::DMem *s,int N,float value) {
 size_t g[]={size_t((N+255)/256)*256},l[]={256}; void *p[]={s,&N,&value};
 int info[]={iris_rw,sizeof(N),sizeof(value)}; run("kernel_reset_f",1,g,l,3,p,info);
}
static void launch_metro(int N,int L,iris::DMem *s,iris::DMem *H,float h,float B,iris::DMem *a,iris::DMem *b,int alt) {
 size_t l[]={BX,BY/2,BZ},g[]={size_t((L+BX-1)/BX)*BX,size_t((L+BY-1)/(2*BY))*(BY/2),size_t((L+BZ-1)/BZ)*BZ};
 void *p[]={&N,&L,s,H,&h,&B,a,b,&alt};
 int info[]={sizeof(N),sizeof(L),iris_rw,iris_r,sizeof(h),sizeof(B),iris_rw,iris_rw,sizeof(alt)};
 run("kernel_metropolis",3,g,l,9,p,info);
}
static void launch_energy(iris::DMem *s,int L,iris::DMem *E,iris::DMem *H,float h,int k) {
 size_t l[]={BX,BY,BZ},g[]={size_t((L+BX-1)/BX)*BX,size_t((L+BY-1)/BY)*BY,size_t((L+BZ-1)/BZ)*BZ};
 void *p[]={s,&L,E,H,&h,&k}; int info[]={iris_r,sizeof(L),iris_rw,iris_r,sizeof(h),sizeof(k)};
 run("kernel_redenergy",3,g,l,6,p,info);
}
