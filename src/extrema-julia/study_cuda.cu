#include <cuda_runtime.h>
#include "../extrema-cuda/kernels.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdint>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(3);}}while(0)
static_assert(sizeof(bool)==1,"Boolartifactrequiresbytepayload");
std::vector<uint64_t>load(std::string f){std::ifstream s(f,std::ios::binary|std::ios::ate);size_t n=s.tellg();s.seekg(0);std::vector<uint64_t>x(n/8);s.read((char*)x.data(),n);return x;}
struct Input{int ty,dim,nx,ny,n;virtual void launch(int,bool,int)=0;virtual std::vector<uint64_t>echo()=0;virtual std::vector<uint64_t>results()=0;virtual void reset()=0;};
template<class T>struct Data:Input{T*base;uint8_t*r;Data(int t,int d,int x,int y,const std::vector<uint64_t>&bits){ty=t;dim=d;nx=x;ny=y;n=x*y;std::vector<uint8_t>bytes((n+16)*sizeof(T),0xA5);for(int i=0;i<n;i++)memcpy(bytes.data()+(i+8)*sizeof(T),&bits[i],sizeof(T));CK(cudaMalloc(&base,bytes.size()));CK(cudaMemcpy(base,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));CK(cudaMalloc(&r,n+16));}
 void launch(int order,bool clip,int axis){if(dim==1)relextrema_1D<T><<<(n+255)/256,256>>>(n,order,clip,base+8,(bool*)(r+8));else relextrema_2D<T><<<dim3((nx+15)/16,(ny+15)/16),dim3(16,16)>>>(nx,ny,order,clip,axis,base+8,(bool*)(r+8));}
 std::vector<uint64_t>echo(){std::vector<uint8_t>b((n+16)*sizeof(T));CK(cudaMemcpy(b.data(),base,b.size(),cudaMemcpyDeviceToHost));std::vector<uint64_t>out(n+16,0);for(int i=0;i<n+16;i++)memcpy(&out[i],b.data()+i*sizeof(T),sizeof(T));return out;}
 std::vector<uint64_t>results(){std::vector<uint8_t>b(n+16);CK(cudaMemcpy(b.data(),r,b.size(),cudaMemcpyDeviceToHost));return std::vector<uint64_t>(b.begin(),b.end());}
 void reset(){CK(cudaMemset(r,165,n+16));}
};
struct Path{Input*a;int order;bool clip;int axis;};
int main(){std::string root=std::string(__FILE__).substr(0,std::string(__FILE__).find_last_of('/'));auto fixture=load(root+"/fixture.bin"),gold=load(root+"/oracle.bin");std::vector<Input*>arrays;std::vector<Path>paths;size_t pos=0;for(int ai=0;ai<12;ai++){int ty=fixture[pos++],dim=fixture[pos++],nx=fixture[pos++],ny=fixture[pos++],n=nx*ny;std::vector<uint64_t>bits(fixture.begin()+pos,fixture.begin()+pos+n);pos+=n;Input*a=nullptr;switch(ty){case 1:a=new Data<int32_t>(ty,dim,nx,ny,bits);break;case 2:a=new Data<int64_t>(ty,dim,nx,ny,bits);break;case 3:a=new Data<float>(ty,dim,nx,ny,bits);break;case 4:a=new Data<double>(ty,dim,nx,ny,bits);break;}arrays.push_back(a);
 for(int axis:dim==1?std::vector<int>{1}:std::vector<int>{0,1}){if(nx==1){paths.push_back({a,1,true,axis});paths.push_back({a,1,false,axis});}else{for(int o=1;o<=128;o*=2)paths.push_back({a,o,true,axis});for(int o:{1,8,16})paths.push_back({a,o,false,axis});}}
 }for(auto&p:paths)p.a->launch(p.order,p.clip,p.axis);CK(cudaDeviceSynchronize());
 for(int trial=1;trial<=5;trial++){std::vector<uint64_t>prefix,out,counts;std::vector<double>times;double elapsed=0;for(auto*a:arrays){prefix.insert(prefix.end(),{uint64_t(a->ty),uint64_t(a->dim),uint64_t(a->nx),uint64_t(a->ny)});auto z=a->echo();prefix.insert(prefix.end(),z.begin()+8,z.end()-8);}
 for(auto&p:paths){p.a->reset();CK(cudaDeviceSynchronize());auto stamp=std::chrono::steady_clock::now();for(int rep=0;rep<100;rep++)p.a->launch(p.order,p.clip,p.axis);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-stamp).count()/100;elapsed+=dt;times.push_back(dt);auto r=p.a->results(),e=p.a->echo();uint64_t ntrue=0;for(size_t i=8;i<r.size()-8;i++)ntrue+=r[i]==1;counts.push_back(ntrue);out.insert(out.end(),r.begin(),r.end());out.insert(out.end(),e.begin(),e.end());}
 size_t mismatches=0;for(size_t i=0;i<out.size();i++)mismatches+=out[i]!=gold[i];bool ok=prefix==fixture&&out.size()==gold.size()&&mismatches==0;std::vector<uint64_t>words=prefix;words.insert(words.end(),out.begin(),out.end());std::string file="output_trial"+std::to_string(trial)+".bin";std::ofstream os(file,std::ios::binary);os.write((char*)words.data(),words.size()*8);os.close();std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<words.size()<<",\"max_abs_error\":null,\"max_rel_error\":null,\"artifact_word_mismatches\":"<<mismatches<<",\"path_times_seconds\":[";for(size_t i=0;i<times.size();i++){if(i)std::cout<<",";std::cout<<times[i];}std::cout<<"],\"path_true_counts\":[";for(size_t i=0;i<counts.size();i++){if(i)std::cout<<",";std::cout<<counts[i];}std::cout<<"],\"timing_region\":\"sum148_hostinclusive_100launch_batches_div100\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\""<<file<<"\",\"dtype\":\"uint64\",\"count\":"<<words.size()<<",\"byte_order\":\"little\"}}\n";
 }
}
