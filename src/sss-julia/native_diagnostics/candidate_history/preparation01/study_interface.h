#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
using SSSCallback=int(*)(int,int,int,void*,void*,void*,void*,std::size_t,std::size_t);
void sss_launch(int,int,int,cudaStream_t,void*,void*,void*,std::size_t,std::size_t);
cudaError_t sss_malloc(void**,std::size_t);
cudaError_t sss_free(void*);
void sss_observe_graph(void*);
void sss_observe_results(int,int,const short*);
void sss_observe_data(const float*,int,int);
void sss_observe_iteration(void*,long,float,float,float,int,int,long);
void sss_observe_final(void*,void*,void*,void*,void*,long,long);
void sss_start_timer();void sss_stop_timer();
extern "C" int sss_run(const char*,const char*,SSSCallback,int);
