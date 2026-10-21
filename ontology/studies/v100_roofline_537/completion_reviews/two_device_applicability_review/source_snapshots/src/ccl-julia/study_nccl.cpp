// Narrow MPI/NCCL C ABI. The repeated collective loop is owned by each caller.
#include <mpi.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>
static int rank_,size_,local_;static ncclComm_t comm_;static cudaStream_t stream_;
static std::vector<float> prefix_,outputs_;
static void fail(const char* s){fprintf(stderr,"CCL rank%d: %s\n",rank_,s);fflush(stderr);MPI_Abort(MPI_COMM_WORLD,1);std::abort();}
static void gpu(cudaError_t e){if(e!=cudaSuccess)fail(cudaGetErrorString(e));}
static void mpi(int e){if(e!=MPI_SUCCESS)fail("MPI status failure");}
static void nccl(ncclResult_t e){if(e!=ncclSuccess)fail(ncclGetErrorString(e));}
static float input(int c,int rank,int i){return c==0?0.f:c==1?1.f:float(rank+1)*.5f+float(i%17-8)/32.f;}
static float expected(int c,int i){return c==0?0.f:c==1?2.f:1.5f+float(i%17-8)/16.f;}
extern "C" int ccl_mpi_start(){int n=0;char**v=nullptr;mpi(MPI_Init(&n,&v));mpi(MPI_Comm_rank(MPI_COMM_WORLD,&rank_));mpi(MPI_Comm_size(MPI_COMM_WORLD,&size_));MPI_Comm l;mpi(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank_,MPI_INFO_NULL,&l));mpi(MPI_Comm_rank(l,&local_));mpi(MPI_Comm_free(&l));if(size_!=2||local_>1)fail("Expected exactly two local ranks");const char*reg=std::getenv("HECBENCH_CCL_PID_REGISTRY");if(!reg)fail("Missing owned rank PID registry");std::ofstream f(std::string(reg)+"/rank"+std::to_string(rank_)+".pid");f<<getpid();f.close();return local_;}
extern "C" void ccl_comm_start(){int dev;gpu(cudaGetDevice(&dev));if(dev!=local_)fail("wrong rank-local device");gpu(cudaStreamCreate(&stream_));ncclUniqueId id;if(rank_==0)nccl(ncclGetUniqueId(&id));mpi(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD));nccl(ncclCommInitRank(&comm_,size_,id,rank_));int nr,nrank,ndev;nccl(ncclCommCount(comm_,&nr));nccl(ncclCommUserRank(comm_,&nrank));nccl(ncclCommCuDevice(comm_,&ndev));if(nr!=2||nrank!=rank_||ndev!=local_)fail("NCCL communicator metadata");int ver;nccl(ncclGetVersion(&ver));Dl_info info{};if(!dladdr(dlsym(RTLD_DEFAULT,"ncclAllReduce"),&info))fail("NCCL loaded DSO lookup");printf("CCL_RUNTIME rank=%d local=%d pid=%d nccl=%d dso=%s\n",rank_,local_,getpid(),ver,info.dli_fname);fflush(stdout);}
extern "C" void ccl_fill(float*s,float*r,int n,int c){for(int i=0;i<n+2;i++){s[i]=r[i]=4096.25f;}for(int i=0;i<n;i++){s[i+1]=input(c,rank_,i);r[i+1]=-777.f;}}
extern "C" void ccl_validate(const void*s,void*r){cudaPointerAttributes a,b;gpu(cudaPointerGetAttributes(&a,s));gpu(cudaPointerGetAttributes(&b,r));if(a.device!=local_||b.device!=local_)fail("collective pointer device");}
extern "C" void ccl_step(const void*s,void*r,int n){nccl(ncclAllReduce(s,r,n,ncclFloat,ncclSum,comm_,stream_));}
extern "C" double ccl_report(int trial,int c,double dt){printf("CCL_CASE trial=%d case=%d rank=%d seconds_per_collective=%.12g\n",trial,c,rank_,dt);fflush(stdout);double y;mpi(MPI_Allreduce(&dt,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));return y;}
extern "C" void ccl_wait(){gpu(cudaStreamSynchronize(stream_));}
extern "C" void ccl_setup_wait(){gpu(cudaDeviceSynchronize());}
extern "C" double ccl_now(){return MPI_Wtime();}
extern "C" void ccl_barrier(){mpi(MPI_Barrier(MPI_COMM_WORLD));}
extern "C" double ccl_max(double x){double y;mpi(MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));return y;}
extern "C" void ccl_begin_trial(){prefix_.clear();outputs_.clear();}
extern "C" void ccl_capture(const void* s,const void*r,int n,int c,int after){std::vector<float> v(2*(n+2));gpu(cudaMemcpy(v.data(),s,(n+2)*sizeof(float),cudaMemcpyDeviceToHost));gpu(cudaMemcpy(v.data()+n+2,r,(n+2)*sizeof(float),cudaMemcpyDeviceToHost));for(int which=0;which<2;which++){const float*p=v.data()+which*(n+2);if(p[0]!=4096.25f||p[n+1]!=4096.25f)fail("guard changed");for(int i=0;i<n;i++){float want=which?(after?expected(c,i):-777.f):input(c,rank_,i);if(p[i+1]!=want)fail("full independent pointwise checker");}}
std::vector<float> all(rank_==0?v.size()*2:0);mpi(MPI_Gather(v.data(),v.size(),MPI_FLOAT,all.data(),v.size(),MPI_FLOAT,0,MPI_COMM_WORLD));if(rank_==0){auto&out=after?outputs_:prefix_;out.insert(out.end(),all.begin(),all.end());}}
extern "C" void ccl_write_trial(int trial,double seconds){if(rank_!=0)return;if(prefix_.size()!=46137436||outputs_.size()!=46137436)fail("artifact size");std::string name="ccl_trial"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)prefix_.data(),prefix_.size()*4);f.write((char*)outputs_.data(),outputs_.size()*4);f.close();if(!f)fail("artifact write");printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":92274872,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum3_case_maxrank_mean50_NCCL_AllReduce_explicitstream_wait\",\"internal_repeats\":50,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":92274872,\"byte_order\":\"little\"}}\n",trial,seconds,name.c_str());fflush(stdout);}
extern "C" void ccl_finish(){nccl(ncclCommFinalize(comm_));nccl(ncclCommDestroy(comm_));gpu(cudaStreamDestroy(stream_));mpi(MPI_Finalize());}
