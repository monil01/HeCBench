#pragma once
#include <iris/iris.hpp>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
struct Buffer {
 std::vector<unsigned char> host;
 iris::DMem mem;
 explicit Buffer(size_t bytes):host(bytes),mem(host.data(),bytes){}
 void update(const void *src,size_t bytes){if(bytes>host.size())exit(4);std::memcpy(host.data(),src,bytes);mem.update(host.data());}
 void reset(int value,size_t bytes){if(bytes>host.size())exit(4);std::memset(host.data(),value,bytes);mem.update(host.data());}
 void read(void *dst,size_t bytes){if(bytes>host.size())exit(4);iris::Task task;task.flush_out(mem);if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);std::memcpy(dst,host.data(),bytes);}
};
struct Dim3 {size_t x;Dim3(size_t v):x(v){}};
static_assert(sizeof(MINIFE_SCALAR)==8 && sizeof(MINIFE_LOCAL_ORDINAL)==4 && sizeof(MINIFE_GLOBAL_ORDINAL)==4,"Original active double/int/int build required");
inline void launch_waxby_kernel(Dim3 grid,Dim3 block,const int n,const MINIFE_SCALAR alpha,Buffer *xcoefs,const MINIFE_SCALAR beta,Buffer *ycoefs,Buffer *wcoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),const_cast<void *>(static_cast<const void *>(&alpha)),&xcoefs->mem,const_cast<void *>(static_cast<const void *>(&beta)),&ycoefs->mem,&wcoefs->mem};
 int info[]={sizeof(n),sizeof(alpha),iris_r,sizeof(beta),iris_r,iris_rw};
 task.kernel("waxby_kernel",1,nullptr,g,l,6,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_yaxby_kernel(Dim3 grid,Dim3 block,const int n,const MINIFE_SCALAR alpha,Buffer *xcoefs,const MINIFE_SCALAR beta,Buffer *ycoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),const_cast<void *>(static_cast<const void *>(&alpha)),&xcoefs->mem,const_cast<void *>(static_cast<const void *>(&beta)),&ycoefs->mem};
 int info[]={sizeof(n),sizeof(alpha),iris_r,sizeof(beta),iris_rw};
 task.kernel("yaxby_kernel",1,nullptr,g,l,5,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_wxby_kernel(Dim3 grid,Dim3 block,const int n,Buffer *xcoefs,const MINIFE_SCALAR beta,Buffer *ycoefs,Buffer *wcoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),&xcoefs->mem,const_cast<void *>(static_cast<const void *>(&beta)),&ycoefs->mem,&wcoefs->mem};
 int info[]={sizeof(n),iris_r,sizeof(beta),iris_r,iris_rw};
 task.kernel("wxby_kernel",1,nullptr,g,l,5,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_yxby_kernel(Dim3 grid,Dim3 block,const int n,Buffer *xcoefs,const MINIFE_SCALAR beta,Buffer *ycoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),&xcoefs->mem,const_cast<void *>(static_cast<const void *>(&beta)),&ycoefs->mem};
 int info[]={sizeof(n),iris_r,sizeof(beta),iris_rw};
 task.kernel("yxby_kernel",1,nullptr,g,l,4,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_wx_kernel(Dim3 grid,Dim3 block,const int n,Buffer *xcoefs,Buffer *wcoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),&xcoefs->mem,&wcoefs->mem};
 int info[]={sizeof(n),iris_r,iris_rw};
 task.kernel("wx_kernel",1,nullptr,g,l,3,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_dyx_kernel(Dim3 grid,Dim3 block,const int n,Buffer *xcoefs,Buffer *ycoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),&xcoefs->mem,&ycoefs->mem};
 int info[]={sizeof(n),iris_r,iris_rw};
 task.kernel("dyx_kernel",1,nullptr,g,l,3,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_wax_kernel(Dim3 grid,Dim3 block,const int n,const MINIFE_SCALAR alpha,Buffer *xcoefs,Buffer *wcoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),const_cast<void *>(static_cast<const void *>(&alpha)),&xcoefs->mem,&wcoefs->mem};
 int info[]={sizeof(n),sizeof(alpha),iris_r,iris_rw};
 task.kernel("wax_kernel",1,nullptr,g,l,4,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_dyax_kernel(Dim3 grid,Dim3 block,const int n,const MINIFE_SCALAR alpha,Buffer *xcoefs,Buffer *ycoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),const_cast<void *>(static_cast<const void *>(&alpha)),&xcoefs->mem,&ycoefs->mem};
 int info[]={sizeof(n),sizeof(alpha),iris_r,iris_rw};
 task.kernel("dyax_kernel",1,nullptr,g,l,4,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_dot_kernel(Dim3 grid,Dim3 block,const MINIFE_LOCAL_ORDINAL n,Buffer *x,Buffer *y,Buffer *d) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&n)),&x->mem,&y->mem,&d->mem};
 int info[]={sizeof(n),iris_r,iris_r,iris_rw};
 task.kernel("dot_kernel",1,nullptr,g,l,4,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_final_reduce(Dim3 grid,Dim3 block,Buffer *d) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={&d->mem};
 int info[]={iris_rw};
 task.kernel("final_reduce",1,nullptr,g,l,1,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_matvec_kernel(Dim3 grid,Dim3 block,const MINIFE_LOCAL_ORDINAL rows_size,Buffer *Arowoffsets,Buffer *Acols,Buffer *Acoefs,Buffer *xcoefs,Buffer *ycoefs) {
 iris::Task task;size_t g[]={grid.x*block.x},l[]={block.x};
 void *kernel_params[]={const_cast<void *>(static_cast<const void *>(&rows_size)),&Arowoffsets->mem,&Acols->mem,&Acoefs->mem,&xcoefs->mem,&ycoefs->mem};
 int info[]={sizeof(rows_size),iris_r,iris_r,iris_r,iris_r,iris_rw};
 task.kernel("matvec_kernel",1,nullptr,g,l,6,kernel_params,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
