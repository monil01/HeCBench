#pragma once
#include <iris/iris.hpp>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdexcept>
struct Dim { size_t x,y,z; Dim(size_t x_=1,size_t y_=1,size_t z_=1):x(x_),y(y_),z(z_){} };
struct Buffer {void* host; size_t bytes; std::unique_ptr<iris::DMem> mem;};
static std::unordered_map<void*,Buffer> buffers;
static int dmem_allocate(void** ptr,size_t bytes) {
  *ptr=malloc(bytes); if(!*ptr) return -1;
  buffers.emplace(*ptr,Buffer{*ptr,bytes,std::make_unique<iris::DMem>(*ptr,bytes)}); return 0;
}
static int dmem_free(void* ptr) {buffers.erase(ptr); free(ptr); return 0;}
static int dmem_write(void* dst,const void* src,size_t bytes) {
  auto &b=buffers.at(dst); if(bytes>b.bytes) return -1;
  memcpy(dst,src,bytes); return b.mem->update(dst);
}
static int dmem_read(void* dst,void* src,size_t bytes) {
  auto &b=buffers.at(src); if(bytes>b.bytes) return -1;
  iris::Task task; task.flush_out(*b.mem);
  if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return -1;
  memcpy(dst,src,bytes); return 0;
}
struct Parameter {iris::DMem* mem=nullptr; int value=0;};
static void add_arg(std::vector<Parameter>& p,float* ptr) {p.push_back({buffers.at(ptr).mem.get(),0});}
static void add_arg(std::vector<Parameter>& p,clusters_t* c) {
  add_arg(p,c->N);
  add_arg(p,c->pi);
  add_arg(p,c->constant);
  add_arg(p,c->avgvar);
  add_arg(p,c->means);
  add_arg(p,c->R);
  add_arg(p,c->Rinv);
  add_arg(p,c->memberships);
}
static void add_arg(std::vector<Parameter>& p,int value) {p.push_back({nullptr,value});}
template<class... Args> static void gmm_task(const char* name,Dim grid,Dim block,Args... values) {
  std::vector<Parameter> p; (add_arg(p,values),...);
  std::vector<void*> args; std::vector<int> info;
  for(auto& a:p) {args.push_back(a.mem ? (void*)a.mem : (void*)&a.value); info.push_back(a.mem ? iris_rw : sizeof(int));}
  size_t global[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},local[]={block.x,block.y,block.z};
  int dims=grid.z*block.z>1 ? 3 : grid.y*block.y>1 ? 2 : 1;
  iris::Task task;
  if(task.kernel(name,dims,nullptr,global,local,args.size(),args.data(),info.data())!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error(name);
}
