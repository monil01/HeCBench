#include <iris/iris.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include "reference.hpp"
static int gpu_device=-1;
struct Param {void* ptr;int info;};
static Param param(iris::DMem& mem){return {&mem,iris_rw};}
template<class T> static Param param(const T& value){return {const_cast<T*>(&value),sizeof(T)};}
template<class... Args> static void launch(const char* name,size_t grid,size_t block,Args&&... args) {
  if(block<1 || block>1024) throw std::runtime_error("block size must be 1..1024");
  Param list[]={param(args)...};void* p[sizeof...(args)];int info[sizeof...(args)];
  for(size_t i=0;i<sizeof...(args);++i){p[i]=list[i].ptr;info[i]=list[i].info;}
  size_t g[]={grid*block},l[]={block};iris::Task task;
  if(task.kernel(name,1,nullptr,g,l,sizeof...(args),p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("kernel failed");
}
static void fetch(iris::DMem& mem,void* host) {
  iris::Task task;task.flush_out(mem);
  if(task.submit(gpu_device,nullptr,1) || mem.fetch(host)) throw std::runtime_error("fetch failed");
}

static void anchors() {
  if(reference(14,0xffffffff,0xfffefffe)!=0xfffefffe ||
     reference(12,0xffffffff,0x01010101)!=0xffffffff ||
     reference(40,0x7fff7fff,0x00010001,0)!=0 ||
     reference(91,0xffffffff,0)!=1020) throw std::runtime_error("reference anchors failed");
}
static void check_individual() {
  std::vector<uint32_t> triples;
  auto add=[&](uint32_t a,uint32_t b,uint32_t c){triples.push_back(a);triples.push_back(b);triples.push_back(c);};
  // Exhaust every unsigned/signed byte pair, including all ties and extremes.
  for(uint32_t a=0;a<256;++a)for(uint32_t b=0;b<256;++b)
    add(a*0x01010101u,b*0x01010101u,(a^b)*0x01010101u);
  const uint32_t edge[]={0,1,~0u,0x80000000,0x7fffffff,0x80008000,0x7fff7fff,
    0xffff0000,0x0000ffff,0x00010001,0x80007fff,0x7fff8000,0x80808080,0x7f7f7f7f,
    0xff00ff00,0x00ff00ff,0x1234aba5,0xaaaaaaaa,0x55555555};
  for(auto a:edge)for(auto b:edge)for(auto c:edge)add(a,b,c);
  uint32_t state=123;
  auto random=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return state;};
  for(int i=0;i<4096;++i){uint32_t a=random(),b=random(),c=random();add(a,b,c);}
  int n=int(triples.size()/3);
  std::vector<uint32_t> actual(size_t(n)*OP_COUNT*2,0);
  iris::DMem input(triples.data(),triples.size()*sizeof(uint32_t)),output(actual.data(),actual.size()*sizeof(uint32_t));
  launch("individual_checks",(n+255)/256,256,n,input,output);
  fetch(output,actual.data());
  for(int i=0;i<n;++i)for(int op=0;op<OP_COUNT;++op) {
    uint32_t pred=0,expected=reference(op,triples[i*3],triples[i*3+1],triples[i*3+2],&pred);
    size_t offset=(size_t(i)*OP_COUNT+op)*2;
    if(actual[offset]!=expected||actual[offset+1]!=pred) {
      fprintf(stderr,"FAIL individual case=%d op=%d expected=%08x actual=%08x pred=%u/%u\n",
        i,op,expected,actual[offset],pred,actual[offset+1]);throw std::runtime_error("individual intrinsic mismatch");
    }
  }
  printf("PASS all %d intrinsic results and predicates for %d exhaustive-byte/boundary/random triples (%zu fields)\n",
    OP_COUNT,n,actual.size());
}
int main(int argc,char** argv) {
  if(argc!=3){printf("Usage: %s <number of elements> <repeat>\n",argv[0]);return 1;}
  int n=atoi(argv[1]),repeat=atoi(argv[2]);if(n<1||repeat<1)return 1;
  std::vector<uint32_t> input(n),output(n,0);
  for(int i=0;i<n;++i)input[i]=0x1234aba5 ^ ((i<n/2)?(-i):i);
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;

  gpu_device=device;
  iris::DMem d_input(input.data(),input.size()*sizeof(uint32_t)),d_output(output.data(),output.size()*sizeof(uint32_t));
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i)launch("simd_intrinsics",(n+255)/256,256,n,d_input,d_output);
  auto end=std::chrono::steady_clock::now();
  auto time=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  printf("Average execution time of the SIMD intrinsics kernel: %f (us)\n",(time*1e-3f)/repeat);
  fetch(d_output,output.data());
  uint32_t checksum=0;
  anchors();
  for(int i=0;i<n;++i) {
    uint32_t expected=reference_full(input[i],i);
    if(output[i]!=expected){fprintf(stderr,"FAIL i=%d actual=%08x expected=%08x\n",i,output[i],expected);return 3;}
    checksum^=output[i];
  }
  printf("Checksum = %x\n",checksum);
  printf("PASS full CUDA-derived reference: %d/%d output words exact\n",n,n);
  check_individual();
  platform.finalize();
  return 0;
}
