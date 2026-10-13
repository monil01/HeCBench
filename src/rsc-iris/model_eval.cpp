#include "kernel.h"
#include <cstdlib>
void call_RANSAC_kernel_block(int blocks,int threads,iris::DMem& model_param_local,iris::DMem& flowvectors,
 int flowvector_count,int max_iter,int error_threshold,float convergence_threshold,
 iris::DMem& g_out_id,iris::DMem& model_candidate,iris::DMem& outliers_candidate,int l_mem_size) {
 void* args[]={&model_param_local,&flowvectors,&flowvector_count,&max_iter,&error_threshold,&convergence_threshold,
 &g_out_id,&model_candidate,&outliers_candidate};
 int info[]={iris_r,iris_r,sizeof(flowvector_count),sizeof(max_iter),sizeof(error_threshold),sizeof(convergence_threshold),
 iris_rw,iris_rw,iris_rw};size_t gws[]={size_t(blocks)*size_t(threads)},lws[]={size_t(threads)};
 iris::Task task;
 if(task.kernel("RANSAC_kernel_block",1,nullptr,gws,lws,9,args,info)!=IRIS_SUCCESS ||
 task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}

