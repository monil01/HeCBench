#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <mpi.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <climits>

static void require(bool ok,const char* what){if(!ok){fprintf(stderr,"%s failed\n",what);MPI_Abort(MPI_COMM_WORLD,1);}}
struct Step{int operation,rank,world;size_t send_offset,recv_offset,send_count,recv_count;};
static int communicate(void* rawstream,unsigned,void* const* args){
 auto* step=hecbench_vendor_arg<Step*>(args,1);
 float* output=hecbench_vendor_arg<float*>(args,2);float* buffer=hecbench_vendor_arg<float*>(args,3);
 auto stream=static_cast<hipStream_t>(rawstream);
 hipPointerAttribute_t attributes{};
 if(hipPointerGetAttributes(&attributes,output)!=hipSuccess || attributes.type!=hipMemoryTypeDevice)return 1;
 static thread_local bool reported=false;if(!reported){printf("MPI callback verified DMem device pointer on HIP GPU%d\n",attributes.device);reported=true;}
 if(hipStreamSynchronize(stream)!=hipSuccess)return 1;
 int left=(step->rank-1+step->world)%step->world,right=(step->rank+1)%step->world;
 MPI_Status status;int error;
 if(step->operation==0){
  MPI_Request request;
  error=MPI_Irecv(buffer,int(step->recv_count),MPI_FLOAT,left,0,MPI_COMM_WORLD,&request);if(error)return error;
  error=MPI_Send(output+step->send_offset,int(step->send_count),MPI_FLOAT,right,0,MPI_COMM_WORLD);if(error)return error;
  return MPI_Wait(&request,&status);
 }
 return MPI_Sendrecv(output+step->send_offset,int(step->send_count),MPI_FLOAT,right,0,
                     output+step->recv_offset,int(step->recv_count),MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
}
static void exchange(iris::DMem& output,iris::DMem& buffer,Step& step,int device){
 hecbench_vendor_callback callback=communicate;Step* info=&step;size_t one=1;
 void* args[]={&callback,&info,&output,&buffer};int modes[]={sizeof(callback),sizeof(info),iris_rw,iris_rw};
 iris::Task task;require(task.kernel("hecbench_vendor",1,nullptr,&one,&one,4,args,modes)==IRIS_SUCCESS,"MPI GPU task");
 require(task.submit(device,nullptr,true)==IRIS_SUCCESS,"MPI GPU submit");
}
static void copy_data(iris::DMem& input,iris::DMem& output,size_t n,int device){
 size_t global=32*256,local=256;
 void* args[]={&input,&output,&n};int modes[]={iris_r,iris_w,sizeof(n)};
 iris::Task task;require(task.kernel("copy_data",1,nullptr,&global,&local,3,args,modes)==IRIS_SUCCESS,"copy task");
 require(task.submit(device,nullptr,true)==IRIS_SUCCESS,"copy submit");
}
static void ring(iris::DMem& input,iris::DMem& output,iris::DMem& buffer,size_t n,int device,int rank,int world){
 std::vector<size_t> lengths(world);
 require(MPI_Allgather(&n,1,MPI_UNSIGNED_LONG,lengths.data(),1,MPI_UNSIGNED_LONG,MPI_COMM_WORLD)==MPI_SUCCESS,"length allgather");
 for(auto count:lengths)require(count==n,"equal input lengths");
 std::vector<size_t> sizes(world,n/world),offsets(world);
 for(size_t i=0;i<n%world;++i)++sizes[i];
 for(int i=1;i<world;++i)offsets[i]=offsets[i-1]+sizes[i-1];
 copy_data(input,output,n,device);
 for(int i=0;i<world-1;++i){
  int recv=(rank-i-1+world)%world,send=(rank-i+world)%world;
  Step step{0,rank,world,offsets[send],offsets[recv],sizes[send],sizes[recv]};exchange(output,buffer,step,device);
  size_t global=32*256,local=256,length=sizes[recv],offset=offsets[recv];
  void* args[]={&buffer,&output,&offset,&length};int modes[]={iris_r,iris_rw,sizeof(offset),sizeof(length)};
  iris::Task task;require(task.kernel("kernel_add_segment",1,nullptr,&global,&local,4,args,modes)==IRIS_SUCCESS,"GPU add task");
  require(task.submit(device,nullptr,true)==IRIS_SUCCESS,"GPU add submit");
 }
 for(int i=0;i<world-1;++i){
  int send=(rank-i+1+world)%world,recv=(rank-i+world)%world;
  Step step{1,rank,world,offsets[send],offsets[recv],sizes[send],sizes[recv]};exchange(output,buffer,step,device);
 }
}
int main(int argc,char** argv){
 const char* env=getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!env)env=getenv("SLURM_LOCALID");
 if(!env){fprintf(stderr,"Missing local rank\n");return 1;}int device=atoi(env);
 if(hipSetDevice(device)!=hipSuccess)return 1;
 int provided=0;if(MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided)!=MPI_SUCCESS)return 1;
 require(provided>=MPI_THREAD_MULTIPLE,"MPI_THREAD_MULTIPLE");
 int rank,world;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&world);
 setenv("IRIS_ARCHS","HIP",1);setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",1);
 iris::Platform platform;require(platform.init(&argc,&argv,true)==IRIS_SUCCESS,"IRIS init");
 require(hipSetDevice(device)==hipSuccess,"HIP local context");
 int backend=-1;char name[256]={};iris_device_info(device,iris_backend,&backend,nullptr);iris_device_info(device,iris_name,name,nullptr);
 require(backend==iris_hip,"HIP device");printf("Rank %d IRIS HIP backend=%d device=%s MPI_THREAD_MULTIPLE=%d direct GPU MPI\n",rank,backend,name,provided);
 std::vector<size_t> sizes={0,32,256,1024,4096,16384,65536,262144,1048576,8388608,67108864,536870912};
 std::vector<size_t> iterations={100000,100000,100000,100000,1000,1000,1000,1000,100,50,10,1};
 if(const char* p=getenv("HECBENCH_ALLREDUCE_TEST_LENGTH")){sizes={strtoull(p,nullptr,10)};iterations={100000};}
 const char* cap=getenv("HECBENCH_ALLREDUCE_MAX_ELEMENTS");const char* reps=getenv("HECBENCH_ALLREDUCE_MAX_ITERATIONS");
 const bool patterned=getenv("HECBENCH_ALLREDUCE_PATTERNED")!=nullptr;
 for(size_t c=0;c<sizes.size();++c){
  size_t n=sizes[c],repeat=iterations[c];if(cap&&n>strtoull(cap,nullptr,10))continue;
  if(reps)repeat=std::min(repeat,size_t(strtoull(reps,nullptr,10)));require(repeat>0&&n<=INT_MAX,"validation bounds");
  std::vector<float> input(std::max(n,size_t(1)),1),output(std::max(n,size_t(1))),buffer(std::max(size_t(1),(n+world-1)/world));
  if(patterned)for(size_t j=0;j<n;++j)input[j]=float(int(j%97)-48+11*rank);
  double seconds=0;
  for(size_t iter=0;iter<repeat;++iter){
   iris::DMem in(input.data(),input.size()*sizeof(float)),out(output.data(),output.size()*sizeof(float)),tmp(buffer.data(),buffer.size()*sizeof(float));
   const auto start=std::chrono::steady_clock::now();ring(in,out,tmp,n,device,rank,world);
   seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   // Match native timing: host-visible verification readback is outside the ring timer.
   iris::Task flush;require(flush.flush_out(out)==IRIS_SUCCESS,"flush output");require(flush.submit(device,nullptr,true)==IRIS_SUCCESS,"flush submit");
   for(size_t j=0;j<n;++j){float expected=patterned?float(world*(int(j%97)-48)+11*world*(world-1)/2):float(world);require(output[j]==expected,"whole allreduce output");}
   if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")){char path[4096];snprintf(path,sizeof(path),"%s.rank%d.n%zu.iter%zu.bin",prefix,rank,n,iter);FILE* f=fopen(path,"wb");require(f!=nullptr,"dump open");require(fwrite(output.data(),sizeof(float),n,f)==n,"dump write");fclose(f);}
  }
  printf("Rank %d PASS all %zu outputs x %zu iterations, patterned=%d\n",rank,n,repeat,int(patterned));
  if(rank==0)std::cout<<"Verified allreduce for size "<<n<<" ("<<seconds*1e6/repeat<<" us per iteration)"<<std::endl;
 }
 require(iris_error_count()==0,"IRIS error count");
 MPI_Finalize();platform.finalize();return 0;
}
