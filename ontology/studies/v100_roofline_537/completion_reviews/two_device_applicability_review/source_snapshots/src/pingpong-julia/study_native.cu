#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#define main archived_mpi_main
#define test archived_mpi_test
#include "original_native/main-mpi.cu"
#undef test
#undef main
#define main archived_nccl_main
#define test archived_nccl_test
#include "original_native/main-nccl.cu"
#undef test
#undef main
#include <vector>
#include <cstdlib>
extern "C" {int pp_start();void pp_guard();void pp_comm(void*);void pp_recv(double*,int,int);void pp_send(const double*,int,int);void pp_context_wait();void pp_stream_wait();void pp_barrier();double pp_now();double pp_report(int,int,int,double);void pp_begin();double pp_fixture(int,int);void pp_capture(const double*,int,int);void pp_write(int,double);void pp_finish();}
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::abort();}}
int main(){int rank=pp_start();ck(cudaSetDevice(rank));cudaStream_t stream;ck(cudaStreamCreate(&stream));pp_comm((void*)stream);int sizes[]={65536,262144,1048576,257};
for(int trial=0;trial<=5;trial++){pp_guard();pp_begin();double elapsed=0;
for(int variant=0;variant<2;variant++)for(int count:sizes){pp_guard();std::vector<double>h(count+64,-7777.25);for(int i=0;i<count;i++)h[i+32]=pp_fixture(i,rank);double*raw;ck(cudaMalloc(&raw,h.size()*8));ck(cudaMemcpy(raw,h.data(),h.size()*8,cudaMemcpyHostToDevice));double*d=raw+32;pp_context_wait();ck(cudaMemcpy(h.data(),raw,h.size()*8,cudaMemcpyDeviceToHost));if(trial)pp_capture(h.data(),count,0);
for(int repeat=0;repeat<5;repeat++){if(!rank){pp_send(d,count,variant);pp_recv(d,count,variant);}else{pp_recv(d,count,variant);if(variant==0){pp_context_wait();archived_mpi_test<<<1024,256>>>(d,count);pp_context_wait();}else{archived_nccl_test<<<1024,256,0,stream>>>(d,count);pp_stream_wait();}pp_send(d,count,variant);}}pp_context_wait();ck(cudaMemcpy(h.data(),raw,h.size()*8,cudaMemcpyDeviceToHost));if(trial)pp_capture(h.data(),count,1);
pp_barrier();double start=pp_now();for(int repeat=0;repeat<50;repeat++){if(!rank){pp_send(d,count,variant);pp_recv(d,count,variant);}else{pp_recv(d,count,variant);pp_send(d,count,variant);}}if(variant)pp_stream_wait();else pp_context_wait();double seconds=pp_now()-start;elapsed+=pp_report(trial,variant,count,seconds);pp_context_wait();ck(cudaMemcpy(h.data(),raw,h.size()*8,cudaMemcpyDeviceToHost));if(trial)pp_capture(h.data(),count,2);ck(cudaFree(raw));pp_guard();}
if(trial)pp_write(trial,elapsed);pp_barrier();}pp_finish();ck(cudaStreamDestroy(stream));}
