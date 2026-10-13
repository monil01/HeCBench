#include "kernel.h"
#include <cstdlib>
void call_StreamCompaction_kernel(int blocks,int threads,int size,T value,int n_tasks,float alpha,
 iris::DMem& output,iris::DMem& input,iris::DMem& flags,int l_mem_size) {
 void* args[]={&size,&value,&n_tasks,&alpha,&output,&input,&flags};
 int info[]={sizeof(size),sizeof(value),sizeof(n_tasks),sizeof(alpha),iris_rw,iris_r,iris_rw};
 size_t gws[]={size_t(blocks)*size_t(threads)},lws[]={size_t(threads)};
 iris::Task task;
 if(task.kernel("StreamCompaction",1,nullptr,gws,lws,7,args,info)!=IRIS_SUCCESS ||
 task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}

