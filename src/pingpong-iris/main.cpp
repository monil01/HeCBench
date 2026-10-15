#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime_api.h>
#include <mpi.h>
#include <rccl/rccl.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <algorithm>
#ifndef PINGPONG_RCCL
#define PINGPONG_RCCL 0
#endif
static constexpr bool rccl=PINGPONG_RCCL!=0;
static const char* variant=rccl?"NCCL":"MPI";
static void require(bool ok,const char* what){if(!ok){fprintf(stderr,"%s failed\n",what);MPI_Abort(MPI_COMM_WORLD,1);}}
struct Exchange{int rank,operation,repeat;long n;ncclComm_t comm;};
static int communicate(void* rawstream,unsigned,void* const* args){
 auto* state=hecbench_vendor_arg<Exchange*>(args,1);auto data=hecbench_vendor_arg<double*>(args,2);
 auto stream=static_cast<hipStream_t>(rawstream);hipPointerAttribute_t attr{};
 if(hipPointerGetAttributes(&attr,data)!=hipSuccess||attr.type!=hipMemoryTypeDevice)return 1;
 static thread_local bool reported=false;if(!reported){printf("%s callback verified DMem GPU%d pointer\n",variant,attr.device);reported=true;}
 if(hipStreamSynchronize(stream)!=hipSuccess)return 1;
 const auto transfer=[&](bool send)->int{
  if constexpr(rccl){
   ncclResult_t status=send?ncclSend(data,state->n,ncclFloat64,1-state->rank,state->comm,stream):ncclRecv(data,state->n,ncclFloat64,1-state->rank,state->comm,stream);
   return int(status);
  }else{
   MPI_Status status;int tag=send?(state->rank==0?10:20):(state->rank==0?20:10);
   return send?MPI_Send(data,int(state->n),MPI_DOUBLE,1-state->rank,tag,MPI_COMM_WORLD):MPI_Recv(data,int(state->n),MPI_DOUBLE,1-state->rank,tag,MPI_COMM_WORLD,&status);
  }
 };
 if(state->operation==2){
  for(int i=0;i<state->repeat;++i){int status=transfer(state->rank==0);if(status)return status;status=transfer(state->rank!=0);if(status)return status;}
 }else{int status=transfer(state->operation==0);if(status)return status;}
 return int(hipStreamSynchronize(stream));
}
static void exchange(iris::DMem& memory,Exchange& state,int device){
 hecbench_vendor_callback callback=communicate;Exchange* ptr=&state;size_t one=1;
 void* args[]={&callback,&ptr,&memory};int modes[]={sizeof(callback),sizeof(ptr),iris_rw};iris::Task task;
 require(task.kernel("hecbench_vendor",1,nullptr,&one,&one,3,args,modes)==IRIS_SUCCESS,"transfer task");require(task.submit(device,nullptr,true)==IRIS_SUCCESS,"transfer submit");
}
static void increment(iris::DMem& memory,long n,int device){
 size_t global=1024*256,local=256;void* args[]={&memory,&n};int modes[]={iris_rw,sizeof(n)};iris::Task task;
 require(task.kernel("pingpong_increment",1,nullptr,&global,&local,2,args,modes)==IRIS_SUCCESS,"increment task");require(task.submit(device,nullptr,true)==IRIS_SUCCESS,"increment submit");
}
static void observe(iris::DMem& memory,std::vector<double>& values,int rank,const char* stage,int device,bool pattern){
 iris::Task fetch;require(fetch.flush_out(memory)==IRIS_SUCCESS,"fetch");require(fetch.submit(device,nullptr,true)==IRIS_SUCCESS,"fetch submit");
 for(size_t i=0;i<values.size();++i)require(values[i]==double((pattern?int(i%97)-48:0)+5),"whole pingpong output");
 printf("%s Rank %d PASS all %zu %s outputs\n",variant,rank,values.size(),stage);
 if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")){char path[4096];snprintf(path,sizeof(path),"%s.rank%d.n%zu.%s.bin",prefix,rank,values.size(),stage);FILE* f=fopen(path,"wb");require(f!=nullptr,"dump open");require(fwrite(values.data(),sizeof(double),values.size(),f)==values.size(),"dump write");fclose(f);}
}
int main(int argc,char** argv){
 const char* local=getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)local=getenv("SLURM_LOCALID");if(!local)return 1;int device=atoi(local);
 if(hipSetDevice(device)!=hipSuccess)return 1;int provided=0;if(MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided)!=MPI_SUCCESS)return 1;
 require(provided>=MPI_THREAD_MULTIPLE,"MPI_THREAD_MULTIPLE");int world,rank;MPI_Comm_size(MPI_COMM_WORLD,&world);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 if(world!=2){if(rank==0)printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n",world);MPI_Finalize();return 0;}
 setenv("IRIS_ARCHS","HIP",1);setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",1);iris::Platform platform;require(platform.init(&argc,&argv,true)==IRIS_SUCCESS,"IRIS init");require(hipSetDevice(device)==hipSuccess,"HIP device");
 int backend=-1;char name[256]={};iris_device_info(device,iris_backend,&backend,nullptr);iris_device_info(device,iris_name,name,nullptr);require(backend==iris_hip,"HIP backend");printf("%s Rank%d IRIS HIP backend=%d device=%s MPI_THREAD_MULTIPLE=%d\n",variant,rank,backend,name,provided);
 ncclComm_t comm=nullptr;if constexpr(rccl){ncclUniqueId id;if(rank==0)require(ncclGetUniqueId(&id)==ncclSuccess,"RCCL ID");require(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD)==MPI_SUCCESS,"ID broadcast");require(ncclCommInitRank(&comm,world,id,rank)==ncclSuccess,"RCCL communicator");}
 const char* cap=getenv("HECBENCH_PINGPONG_MAX_EXP");int maximum=cap?std::min(27,atoi(cap)):27;
 const char* custom=getenv("HECBENCH_PINGPONG_TEST_LENGTH");const bool patterned=getenv("HECBENCH_PINGPONG_PATTERNED")!=nullptr;
 int repeat=50;if(const char* p=getenv("HECBENCH_PINGPONG_MAX_ITERATIONS"))repeat=std::min(repeat,atoi(p));require(repeat>0,"iterations");
 for(int exponent=16;exponent<=maximum;++exponent){
  if(custom&&exponent>16)break;long n=custom?strtol(custom,nullptr,10):(1L<<exponent);require(n>0&&n<=INT_MAX,"length");
  std::vector<double> values(n,0);if(patterned)for(long i=0;i<n;++i)values[i]=double(int(i%97)-48+7*rank);
  iris::DMem memory(values.data(),values.size()*sizeof(double));Exchange state{rank,0,repeat,n,comm};
  for(int warm=0;warm<5;++warm){
   if(rank==0){state.operation=0;exchange(memory,state,device);state.operation=1;exchange(memory,state,device);}
   else{state.operation=1;exchange(memory,state,device);increment(memory,n,device);state.operation=0;exchange(memory,state,device);}
  }
  observe(memory,values,rank,"warm",device,patterned);
  state.operation=2;double start=MPI_Wtime();exchange(memory,state,device);double elapsed=MPI_Wtime()-start;
  double seconds=elapsed/(2.*repeat);if(rank==0)printf("%s%s Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",variant,rccl?":":" :",8*n,seconds,(double(8*n)/1.e9)/seconds);
  observe(memory,values,rank,"timed",device,patterned);
 }
 if constexpr(rccl){require(ncclCommFinalize(comm)==ncclSuccess,"RCCL finalize");require(ncclCommDestroy(comm)==ncclSuccess,"RCCL destroy");}
 require(iris_error_count()==0,"IRIS error count");MPI_Finalize();platform.finalize();return 0;
}
