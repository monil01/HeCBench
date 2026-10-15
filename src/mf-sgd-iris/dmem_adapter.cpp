#include <iris/iris.hpp>
#include "dmem_adapter.hpp"
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace {
struct Allocation {
 void* host;
 std::size_t bytes;
 std::unique_ptr<iris::DMem> memory;
 Allocation(std::size_t n):host(std::calloc(1,n)),bytes(n){if(!host)throw std::bad_alloc();memory.reset(new iris::DMem(host,n));}
 ~Allocation(){memory.reset();std::free(host);}
};
std::unordered_map<const void*,std::unique_ptr<Allocation>> allocations;
std::unique_ptr<iris::Platform> platform;
int selected=0;
Allocation& allocation(const void*p){auto it=allocations.find(p);if(it==allocations.end())throw std::runtime_error("unknown DMem allocation");return *it->second;}
void* mem(const void*p){return allocation(p).memory.get();}
void require(int code){if(code!=IRIS_SUCCESS)throw std::runtime_error("IRIS DMem task failure");}
void launch(const char*name,std::size_t grid,std::size_t block,std::vector<void*>args,std::vector<int>info,std::initializer_list<const void*>outputs){
 if(!grid||!block)throw std::runtime_error("invalid source launch geometry");
 std::size_t global[]={grid*block},local[]={block};iris::Task task;
 require(task.kernel(name,1,nullptr,global,local,args.size(),args.data(),info.data()));
 for(auto*p:outputs)require(task.flush_out(*allocation(p).memory));
 require(task.submit(selected,nullptr,1));
}
}
void dmem_platform_init(int*argc,char***argv){platform.reset(new iris::Platform);require(platform->init(argc,argv,1));}
void dmem_platform_finalize(){allocations.clear();require(platform->finalize());platform.reset();}
int dmemGetDeviceCount(int*n){return iris_device_count(n);}
int dmemSetDevice(int n){selected=n;return 0;}
int dmemMalloc(void**p,std::size_t n){auto a=std::make_unique<Allocation>(n);*p=a->host;allocations.emplace(*p,std::move(a));return 0;}
int dmemMallocHost(void**p,std::size_t n){*p=std::malloc(n);return *p?0:1;}
int dmemFree(void*p){return allocations.erase(p)==1?0:1;}
int dmemFreeHost(void*p){std::free(p);return 0;}
int dmemMemcpy(void*dst,const void*src,std::size_t bytes,dmemMemcpyKind kind){
 if(kind==dmemMemcpyHostToDevice){auto&a=allocation(dst);if(bytes>a.bytes)throw std::runtime_error("DMem upload bounds");std::memcpy(a.host,src,bytes);require(a.memory->update(a.host));}
 else {auto&a=allocation(src);if(bytes>a.bytes)throw std::runtime_error("DMem fetch bounds");std::memcpy(dst,a.host,bytes);}
 return 0;
}
int dmemMemcpyAsync(void*d,const void*s,std::size_t n,dmemMemcpyKind k,dmemStream_t){return dmemMemcpy(d,s,n,k);}
int dmemDeviceSynchronize(){return iris_error_count()?1:0;}
int dmemPeekAtLastError(){return dmemDeviceSynchronize();}
int dmemStreamCreate(dmemStream_t*s){static int token=0;*s=++token;return 0;}
const char*dmemGetErrorString(int code){return code?"IRIS DMem error":"success";}
void launch_init_rand_state(std::size_t grid,std::size_t block,unsigned seed,unsigned*state){launch("init_random_full",grid,block,{&seed,mem(state)},{sizeof(seed),iris_w},{state});}
void launch_init_rand_state(std::size_t grid,std::size_t block,unsigned seed,unsigned*state,int n){launch("init_random_bounded",grid,block,{&seed,mem(state),&n},{sizeof(seed),iris_w,sizeof(n)},{state});}
void launch_random_init(std::size_t grid,std::size_t block,unsigned*state,int ns,half*array,long long n,long long k,float scale){launch("random_init",grid,block,{mem(state),&ns,mem(array),&n,&k,&scale},{iris_rw,sizeof(ns),iris_w,sizeof(n),sizeof(k),sizeof(scale)},{state,array});}
void launch_transform_half(std::size_t grid,std::size_t block,const half*input,float*output,long long n){launch("transform_half",grid,block,{mem(input),mem(output),&n},{iris_r,iris_w,sizeof(n)},{output});}
void launch_sgd_k128_kernel_hogwild_warp32_lrate(std::size_t grid,std::size_t block,const mf_node*R,long long nnz,half*p,half*q,unsigned*state,const float*rates,long long us,long long vs,int k,int iters,int iter,int update,int count,int vector_size,float lp,float lq,int ug,int vg,int uid,int vid){
 launch("sgd_k128_kernel_hogwild_warp32_lrate",grid,block,{mem(R),&nnz,mem(p),mem(q),mem(state),mem(rates),&us,&vs,&k,&iters,&iter,&update,&count,&vector_size,&lp,&lq,&ug,&vg,&uid,&vid},{iris_r,sizeof(nnz),iris_rw,iris_rw,iris_rw,iris_r,sizeof(us),sizeof(vs),sizeof(k),sizeof(iters),sizeof(iter),sizeof(update),sizeof(count),sizeof(vector_size),sizeof(lp),sizeof(lq),sizeof(ug),sizeof(vg),sizeof(uid),sizeof(vid)},{p,q,state});
}
