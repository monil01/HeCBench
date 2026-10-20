#pragma once
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstdint>
namespace tsne_guards {
inline constexpr size_t G=64;
struct Allocation{void*raw;size_t bytes,element_bytes;};
struct Check{size_t sequence,bytes,element_bytes;std::vector<unsigned char>head,tail;bool pass;};
inline std::unordered_map<void*,Allocation>live;
inline std::vector<Check>checks;
inline bool ok=true;
inline double observation_seconds=0;
inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline void reset(){if(!live.empty())throw std::runtime_error("Live guarded allocation before trial");checks.clear();ok=true;observation_seconds=0;}
inline void save(const std::string&path){std::ofstream f(path);f<<"{\"scope\":\"All explicit source Thrust device_vector allocations, not Thrust/CUDA internal temporary pools or CPU memory\",\"pass\":"<<(ok&&live.empty()?"true":"false")<<",\"live_allocations\":"<<live.size()<<",\"allocations\":[";for(size_t i=0;i<checks.size();i++){auto&q=checks[i];if(i)f<<",";f<<"{\"sequence\":"<<q.sequence<<",\"payload_bytes\":"<<q.bytes<<",\"element_bytes\":"<<q.element_bytes<<",\"guard_elements_each_end\":64,\"expected_byte\":165,\"pass\":"<<(q.pass?"true":"false")<<",\"head_bytes\":[";for(size_t j=0;j<q.head.size();j++)f<<(j?",":"")<<unsigned(q.head[j]);f<<"],\"tail_bytes\":[";for(size_t j=0;j<q.tail.size();j++)f<<(j?",":"")<<unsigned(q.tail[j]);f<<"]}";}f<<"]}\n";if(!f)throw std::runtime_error("Guard evidence write failed");}
}
template<class T>struct TSNEGuardAllocator:thrust::device_allocator<T>{
 using Base=thrust::device_allocator<T>;using value_type=T;using pointer=typename Base::pointer;
 template<class U>struct rebind{using other=TSNEGuardAllocator<U>;};
 TSNEGuardAllocator()=default;template<class U>TSNEGuardAllocator(const TSNEGuardAllocator<U>&){}
 pointer allocate(size_t n){auto raw=Base::allocate(n+2*tsne_guards::G);auto p=raw+tsne_guards::G;void*address=thrust::raw_pointer_cast(p);size_t g=tsne_guards::G*sizeof(T);tsne_guards::ck(cudaMemset(thrust::raw_pointer_cast(raw),165,g));tsne_guards::ck(cudaMemset(thrust::raw_pointer_cast(p+n),165,g));if(!tsne_guards::live.emplace(address,tsne_guards::Allocation{thrust::raw_pointer_cast(raw),n*sizeof(T),sizeof(T)}).second)throw std::runtime_error("Duplicate live guard allocation");return p;}
 void deallocate(pointer p,size_t n){tsne_guards::ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();auto it=tsne_guards::live.find(thrust::raw_pointer_cast(p));if(it==tsne_guards::live.end()||it->second.bytes!=n*sizeof(T))throw std::runtime_error("Unknown/mismatched guard free");size_t g=tsne_guards::G*sizeof(T);tsne_guards::Check q{tsne_guards::checks.size(),n*sizeof(T),sizeof(T),std::vector<unsigned char>(g),std::vector<unsigned char>(g),true};tsne_guards::ck(cudaMemcpy(q.head.data(),it->second.raw,g,cudaMemcpyDeviceToHost));tsne_guards::ck(cudaMemcpy(q.tail.data(),thrust::raw_pointer_cast(p+n),g,cudaMemcpyDeviceToHost));for(auto x:q.head)q.pass&=x==165;for(auto x:q.tail)q.pass&=x==165;tsne_guards::ok&=q.pass;tsne_guards::checks.push_back(std::move(q));tsne_guards::live.erase(it);tsne_guards::observation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();Base::deallocate(p-tsne_guards::G,n+2*tsne_guards::G);}
};
namespace thrust{template<class T,class Unused=TSNEGuardAllocator<T>>using guarded_device_vector=device_vector<T,TSNEGuardAllocator<T>>;}
// Applied consistently to original declarations/definitions in every source TU.
#define device_vector guarded_device_vector
