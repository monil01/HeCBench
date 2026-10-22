#include <cuda_runtime.h>
#include <mpi.h>
#include <cublas_v2.h>
#include <map>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <cstring>
static std::map<void*,std::pair<unsigned char*,size_t>>allocations;
static cudaError_t guardedMalloc(void**p,size_t n){unsigned char*raw;cudaError_t e=cudaMalloc(&raw,n+512);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));e=cudaMemset(raw,0xa5,n+512);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));e=cudaMemset(raw+256,0,n);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));*p=raw+256;allocations[*p]={raw,n};return cudaSuccess;}
static cudaError_t guardedFree(void*p){auto it=allocations.find(p);if(it==allocations.end())throw std::runtime_error("unknown guarded allocation");auto e=cudaFree(it->second.first);allocations.erase(it);return e;}
#define main archived_original_main
#define cudaMalloc guardedMalloc
#define cudaFree guardedFree
#include "/home/imo/HeCBench_julia/src/sparkler-julia/original_native/main.cu"
#undef cudaFree
#undef cudaMalloc
#undef main
#include "/home/imo/HeCBench_julia/src/sparkler-julia/study_abi.h"
#include "/home/imo/HeCBench_julia/src/sparkler-julia/study_capture.h"
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static std::vector<unsigned char>frame(float*p){auto a=allocations.at(p);std::vector<unsigned char>v(a.second+512);ck(cudaMemcpy(v.data(),a.first,v.size(),cudaMemcpyDeviceToHost));return v;}
static std::vector<unsigned char>guardbytes(float*p){auto a=allocations.at(p);std::vector<unsigned char>v(512);ck(cudaMemcpy(v.data(),a.first,256,cudaMemcpyDeviceToHost));ck(cudaMemcpy(v.data()+256,a.first+256+a.second,256,cudaMemcpyDeviceToHost));return v;}
int main(){try{auto root=std::filesystem::path("/tmp/hecbench-roofline-v100-20261020/batch024-completion-nz-priority/sparkler/cuda/scratch/profile_outputs/private_capture");int rank=study_start("cuda",root.c_str());study_blas_linked_symbol(reinterpret_cast<void*>(cublasSgemm));for(int trial=0;trial<=5;trial++){study_begin_trial(trial);double seconds=0;uint64_t hashes[3]{};double flops[3]{};int ci=0;
for(auto cfg:study_cases()){uint64_t base=cfg.vectors*rank/4,next=cfg.vectors*(rank+1)/4,local=next-base,up=(cfg.vectors+3)/4,m=2*up;cudaStream_t stream;ck(cudaStreamCreate(&stream));auto handle=reinterpret_cast<cublasHandle_t>(study_blas_start((void*)stream));{
Matrix<float>A(m,cfg.fields),B(m,cfg.fields),C(m,m);study_prep_begin((void*)stream);set_input_matrix(A,base,1.f,stream);study_prep_end((void*)stream,ci);C.to_device(stream);ck(cudaStreamSynchronize(stream));ck(cudaDeviceSynchronize());auto v=frame(A.d());study_input_frame(v.data(),v.size(),ci,0,0,0,A.nr(),A.nc(),A.nru(),A.ncu(),base);uint64_t hash=0;double flop=0;
for(int iteration=1;iteration<=cfg.iterations;iteration++)for(int step=1;step<=3;step++){ck(cudaStreamSynchronize(stream));study_barrier();int right=(rank+step-1)%4;uint64_t baser=cfg.vectors*right/4,nextr=cfg.vectors*(right+1)/4,localr=nextr-baser;bool active=step<=(rank<2?3:2);if(active){study_prep_begin((void*)stream);set_input_matrix(B,baser,1.f,stream);study_prep_end((void*)stream,ci);if(iteration==1){v=frame(B.d());study_input_frame(v.data(),v.size(),ci,iteration,step,1,B.nr(),B.nc(),B.nru(),B.ncu(),baser);}}
ck(cudaStreamSynchronize(stream));study_barrier();auto start=std::chrono::steady_clock::now();if(active){perform_gemm<TCSelector<TC_METHOD_FLOAT32>,float,float>(handle,m,m,cfg.fields,A,B,C);flop+=2.*m*m*cfg.fields;}ck(cudaStreamSynchronize(stream));ck(cudaDeviceSynchronize());study_barrier();auto stop=std::chrono::steady_clock::now();seconds+=std::chrono::duration<double>(stop-start).count();
if(active){C.from_device(stream);ck(cudaStreamSynchronize(stream));for(uint64_t c=0;c<m;c+=89)for(uint64_t r=0;r<m;r+=113){uint64_t a=nonzero_stride(r+base),b=nonzero_stride(c+baser),l=lcm(a,b);ASSERT(C.elt(r,c)==1+(cfg.fields-1)/l);}
v=frame(C.d());study_output_frame(v.data(),v.size(),ci,iteration,step,m,C.nru(),C.ncu(),base,baser,cfg.fields);auto ga=guardbytes(A.d()),gb=guardbytes(B.d());study_guard_frame(ga.data(),ci,iteration,step,0);study_guard_frame(gb.data(),ci,iteration,step,1);
for(uint64_t c=0;c<localr;c++){uint64_t cg=c+baser;for(uint64_t r=0;r<local;r++){uint64_t rg=r+base;if(step==1&&r>=c)continue;uint64_t value=C.elt(r,c);hash+=elt_hash(value,rg,cg);}}}}
v=frame(A.d());study_input_frame(v.data(),v.size(),ci,cfg.iterations+1,0,0,A.nr(),A.nc(),A.nru(),A.ncu(),base);int laststep=rank<2?3:2,right=(rank+laststep-1)%4;uint64_t baser=cfg.vectors*right/4;v=frame(B.d());study_input_frame(v.data(),v.size(),ci,cfg.iterations+1,laststep,1,B.nr(),B.nc(),B.nru(),B.ncu(),baser);hashes[ci]=study_sum_u64(hash);flops[ci]=study_sum_f64(flop);if(rank==0)printf("SPARKLER_CASE trial=%d case=%d vectors=%zu fields=%zu iterations=%d hash=%zu flops=%.0f\n",trial,ci,cfg.vectors,cfg.fields,cfg.iterations,hashes[ci],flops[ci]);study_blas_finish(handle);ck(cudaStreamDestroy(stream));}
ci++;}if(trial)study_write_trial(trial,seconds,hashes,flops);study_barrier();}study_finish();}catch(const std::exception&e){fprintf(stderr,"SPARKLER_NATIVE %s\n",e.what());int initialized;MPI_Initialized(&initialized);if(initialized)MPI_Abort(MPI_COMM_WORLD,19);return 2;}}
