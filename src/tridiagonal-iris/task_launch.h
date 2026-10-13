#pragma once
#include <iris/iris.hpp>
#include <vector>
struct dim3{size_t x,y,z;dim3(size_t a,size_t b=1,size_t c=1):x(a),y(b),z(c){}};
inline void launch_task(const char* name,dim3 grid,dim3 block,std::vector<void*> args,std::vector<int> info) {
  iris::Task task;size_t gws[]={grid.x*block.x,grid.y*block.y},lws[]={block.x,block.y};
  if(task.kernel(name,2,nullptr,gws,lws,args.size(),args.data(),info.data())!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(2);
}
