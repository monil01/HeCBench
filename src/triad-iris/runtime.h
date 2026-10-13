#pragma once
#include <iris/iris.hpp>
#include <cstdlib>
#include <cstdio>
#include <cstring>
struct Storage {
 void *host; iris::DMem mem;
 Storage(size_t n):host(calloc(1,n)),mem(host,n) {}
 ~Storage() { free(host); }
};
inline void copy_to(Storage *m,const float *host,size_t n) {
 memcpy(m->host,host,n); m->mem.update(m->host);
}
inline void copy_from(float *host,Storage *m,size_t n) { memcpy(host,m->host,n); }
inline void launch_triad(Storage *A,Storage *B,Storage *C,float scalar,int n) {
 iris::Task task; void *p[]={&A->mem,&B->mem,&C->mem,&scalar};
 int info[]={iris_r,iris_r,iris_w,sizeof(scalar)}; size_t g[]={size_t(n)},l[]={128};
 task.kernel("triad",1,nullptr,g,l,4,p,info); task.flush_out(C->mem);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void prove_backend() {
 int n=0; iris_device_count(&n); bool found=false;
 for(int i=0;i<n;++i) { int b=0,t=0; char name[256]={}; size_t z;
 iris_device_info(i,iris_backend,&b,&z); iris_device_info(i,iris_type,&t,&z); iris_device_info(i,iris_name,name,&z);
 printf("IRIS backend%d/type%d %s\n",b,t,name); if(b==3&&t==256) found=true; }
 if(!found) exit(2);
}
