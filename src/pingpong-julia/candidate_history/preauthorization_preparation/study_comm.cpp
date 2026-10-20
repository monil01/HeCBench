// Narrow individual-operation MPI/NCCL ABI; every repeated roundtrip belongs to caller.
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
static int rank_,pids_[2];static ncclComm_t comm_;static cudaStream_t stream_;
static std::vector<double> prefix_,outputs_;
static void fail(const char*s){fprintf(stderr,"PINGPONG rank%d %s\n",rank_,s);MPI_Abort(MPI_COMM_WORLD,17);std::abort();}
static void m(int x){if(x!=MPI_SUCCESS)fail("MPI status");}static void g(cudaError_t x){if(x!=cudaSuccess)fail(cudaGetErrorString(x));}static void n(ncclResult_t x){if(x!=ncclSuccess)fail(ncclGetErrorString(x));}
extern "C" int pp_start(){m(MPI_Init(nullptr,nullptr));int size;m(MPI_Comm_rank(MPI_COMM_WORLD,&rank_));m(MPI_Comm_size(MPI_COMM_WORLD,&size));if(size!=2)fail("requires two ranks");int pid=getpid();m(MPI_Allgather(&pid,1,MPI_INT,pids_,1,MPI_INT,MPI_COMM_WORLD));const char*r=getenv("PINGPONG_PID_REGISTRY");if(!r)fail("missing PID registry");std::ofstream f(std::string(r)+"/rank"+std::to_string(rank_)+".pid");f<<pid;return rank_;}
extern "C" void pp_guard(){int bad=0;if(rank_==0){FILE*f=popen("nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits","r");if(!f)bad=1;else{char b[256];while(fgets(b,sizeof(b),f)){char*e;long p=strtol(b,&e,10);if(e==b||(p!=pids_[0]&&p!=pids_[1]))bad=1;}if(pclose(f))bad=1;}}m(MPI_Bcast(&bad,1,MPI_INT,0,MPI_COMM_WORLD));if(bad)fail("external GPU5 use; defer");}
extern "C" void pp_comm(void*s){stream_=reinterpret_cast<cudaStream_t>(s);int dev;g(cudaGetDevice(&dev));if(dev!=rank_)fail("rank device mismatch");ncclUniqueId id;if(!rank_)n(ncclGetUniqueId(&id));m(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD));n(ncclCommInitRank(&comm_,2,id,rank_));int v;n(ncclGetVersion(&v));Dl_info info{};dladdr((void*)ncclSend,&info);printf("PINGPONG_RUNTIME rank=%d pid=%d nccl=%d dso=%s stream=%p\n",rank_,getpid(),v,info.dli_fname,(void*)stream_);char lib[MPI_MAX_LIBRARY_VERSION_STRING];int len;m(MPI_Get_library_version(lib,&len));printf("PINGPONG_MPI rank=%d %.*s\n",rank_,len,lib);fflush(stdout);}
extern "C" void pp_recv(double*d,int count,int variant){if(variant==0){MPI_Status s;m(MPI_Recv(d,count,MPI_DOUBLE,1-rank_,rank_?10:20,MPI_COMM_WORLD,&s));}else n(ncclRecv(d,count,ncclFloat64,1-rank_,comm_,stream_));}
extern "C" void pp_send(const double*d,int count,int variant){if(variant==0)m(MPI_Send(d,count,MPI_DOUBLE,1-rank_,rank_?20:10,MPI_COMM_WORLD));else n(ncclSend(d,count,ncclFloat64,1-rank_,comm_,stream_));}
extern "C" void pp_context_wait(){g(cudaDeviceSynchronize());}
extern "C" void pp_stream_wait(){g(cudaStreamSynchronize(stream_));}
extern "C" void pp_barrier(){m(MPI_Barrier(MPI_COMM_WORLD));}
extern "C" double pp_now(){return MPI_Wtime();}
extern "C" double pp_report(int trial,int variant,int count,double seconds){double both[2];m(MPI_Gather(&seconds,1,MPI_DOUBLE,both,1,MPI_DOUBLE,0,MPI_COMM_WORLD));if(!rank_){printf("PINGPONG_CASE trial=%d variant=%d n=%d rank0_seconds_per_transfer=%.17g rank1_seconds_per_transfer=%.17g\n",trial,variant,count,both[0]/100,both[1]/100);fflush(stdout);}return rank_?0:both[0]/100;}
extern "C" void pp_begin(){prefix_.clear();outputs_.clear();}
extern "C" double pp_fixture(int i,int rank){return rank?-3.+double(i%13)/64.:.5+double(i%17-8)/32.;}
extern "C" void pp_capture(const double*raw,int count,int stage){for(int i=0;i<count+64;i++){double want=(i<32||i>=count+32)?-7777.25:(stage?pp_fixture(i-32,0)+5:pp_fixture(i-32,rank_));if(raw[i]!=want)fail("full exact independent value or actual guard mismatch");}std::vector<double>both(rank_?0:2*(count+64));m(MPI_Gather(raw,count+64,MPI_DOUBLE,both.data(),count+64,MPI_DOUBLE,0,MPI_COMM_WORLD));if(!rank_){auto&v=stage?outputs_:prefix_;v.insert(v.end(),both.begin(),both.end());}}
extern "C" void pp_write(int trial,double seconds){if(rank_)return;size_t prefix=4*(65536+262144+1048576+257+4*64);if(prefix_.size()!=prefix||outputs_.size()!=2*prefix)fail("artifact length");std::string path="trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix_.data(),prefix*8);f.write((char*)outputs_.data(),outputs_.size()*8);f.close();if(!f)fail("artifact write");printf("\nSTUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum8_rank0_mean100_oneway_MPI_NCCL_source_pingpong\",\"internal_repeats\":50,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,3*prefix,path.c_str(),3*prefix);fflush(stdout);}
extern "C" void pp_finish(){n(ncclCommFinalize(comm_));n(ncclCommDestroy(comm_));m(MPI_Finalize());}
