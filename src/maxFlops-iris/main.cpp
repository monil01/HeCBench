#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <type_traits>
#include "verification.h"
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

#define BLOCK_SIZE 256
static int device=-1;
template<class T> void test(int repeat,int numFloats){
  std::vector<T> input(numFloats),output(numFloats);
  srand48(123);
  for(int j=0;j<numFloats/2;++j)input[j]=input[numFloats-j-1]=T(drand48()*10.0);
  output=input;
  iris::DMem mem(output.data(),output.size()*sizeof(T));
  size_t gws=numFloats,lws=BLOCK_SIZE;
  const char* families[]={"Add","Mul","MAdd","MulMAdd"};int variants[]={1,2,4,8};
  auto launch=[&](const std::string& name,int family){
    T a=family==1?T(1.01):(family==3?T(3.75):T(10.0));
    T b=family==3?T(0.355):T(0.9899);
    void* args[]={&mem,&repeat,&a,&b};int info[]={iris_rw,sizeof(repeat),sizeof(a),sizeof(b)};
    std::string symbol=name+(std::is_same<T,float>::value?"_float":"_double");
    iris::Task task;check(task.kernel(symbol.c_str(),1,nullptr,&gws,&lws,family<2?3:4,args,info));
    check(task.submit(device,nullptr,true));
  };
  for(int family=0;family<4;++family){
    for(int warmup=0;warmup<4;++warmup)for(int variant:variants)launch(std::string(families[family])+std::to_string(variant),family);
    for(int variant:variants){
      std::string name=std::string(families[family])+std::to_string(variant);
      std::copy(input.begin(),input.end(),output.begin());check(mem.update(output.data()));
      auto start=std::chrono::high_resolution_clock::now();launch(name,family);
      auto end=std::chrono::high_resolution_clock::now();
      printf("kernel execution time (%s): %f (s)\n",name.c_str(),std::chrono::duration<double>(end-start).count());
      iris::Task flush;check(flush.flush_out(mem));check(flush.submit(device,nullptr,true));
      summarize(output.data(),output.size(),name.c_str());
    }
  }
}
int main(int argc,char** argv){
  if(argc!=2&&argc!=3){printf("Usage: %s <repeat> [bounded element count]\n",argv[0]);return 1;}
  int repeat=atoi(argv[1]);int n=argc==3?atoi(argv[2]):2*1024*1024;
  if(n<=0 || n>2*1024*1024 || n%BLOCK_SIZE)return 1;
  iris::Platform platform;check(platform.init(&argc,&argv,true));device=hip_device();
  printf("=== Single-precision floating-point kernels ===\n");test<float>(repeat,n);
  printf("=== Double-precision floating-point kernels ===\n");test<double>(repeat,n);
  check(platform.finalize());return 0;
}
