#include "dmem.h"
#include <chrono>
#include <vector>
#include <utility>
namespace facebook { namespace cuda {
std::vector<float> sort(const std::vector<float>& data,double& time) {
 std::vector<float> out(data.size(),0);
 iris::DMem input(const_cast<float*>(data.data()),data.size()*sizeof(float)),output(out.data(),out.size()*sizeof(float));
 int n=data.size();size_t g[]={32},l[]={32};void* p[]={&input,&output,&n};int info[]={iris_r,iris_w,sizeof(n)};
 iris::Task task;auto start=std::chrono::steady_clock::now();
 if(task.kernel("warp_sort_keys",1,nullptr,g,l,3,p,info)!=IRIS_SUCCESS)throw std::runtime_error("keys kernel");
 task.flush_out(output);if(task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS)throw std::runtime_error("keys task");
 time+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();return out;
}
std::vector<std::pair<float,int>> sortWithIndices(const std::vector<float>& data,double& time) {
 std::vector<float> out(data.size(),0);std::vector<int> index(data.size(),0);
 iris::DMem input(const_cast<float*>(data.data()),data.size()*sizeof(float)),output(out.data(),out.size()*sizeof(float)),indices(index.data(),index.size()*sizeof(int));
 int n=data.size();size_t g[]={32},l[]={32};void* p[]={&input,&output,&indices,&n};int info[]={iris_r,iris_w,iris_w,sizeof(n)};
 iris::Task task;auto start=std::chrono::steady_clock::now();
 if(task.kernel("warp_sort_indices",1,nullptr,g,l,4,p,info)!=IRIS_SUCCESS)throw std::runtime_error("indices kernel");
 task.flush_out(output);task.flush_out(indices);if(task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS)throw std::runtime_error("indices task");
 time+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
 std::vector<std::pair<float,int>> result;for(size_t i=0;i<data.size();++i)result.emplace_back(out[i],index[i]);return result;
}
} }
