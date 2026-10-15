#include <iris/iris.hpp>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <initializer_list>
#include <unistd.h>

template<class T> struct Buffer {
  std::vector<T> host;
  iris::DMem mem;
  Buffer(size_t n):host(n),mem(host.data(),n*sizeof(T)){}
};
static void launch(const char* name,size_t global,size_t local,
 std::initializer_list<void*> args,std::initializer_list<int> info,
 std::initializer_list<iris::DMem*> flush={}) {
  iris::Task task;
  std::vector<void*> a(args); std::vector<int> p(info);
  task.kernel(name,1,nullptr,&global,&local,a.size(),a.data(),p.data());
  for(auto m:flush) task.flush_out(*m);
  if(task.submit(iris_gpu,nullptr,1)) { std::fprintf(stderr,"task %s failed\n",name);std::exit(1); }
}
int main(int argc,char** argv) {
  std::string filename;int option;
  while((option=getopt(argc,argv,"i:h"))!=-1) {
    if(option=='i') filename=optarg;
    else {std::printf("Usage for compression and decompression: ./main -i {inputfile}\n");return option=='h'?0:1;}
  }
  std::ifstream input(filename,std::ios::binary|std::ios::ate);
  auto size=input.tellg();
  if(!input || size<=0 || uint64_t(size)>UINT32_MAX-2048 || size%4) {
    std::fprintf(stderr,"input must contain a nonempty uint32 array of supported size\n");return 1;
  }
  uint32_t bytes=uint32_t(size),blocks=(bytes+2047)/2048,count=blocks+1;
  size_t words=size_t(blocks)*512;
  iris::Platform platform;platform.init(&argc,&argv,1);
  Buffer<uint32_t> in(words),out(words),flagsizes(count),sizes(count),foff(count),offsets(count);
  Buffer<uint8_t> tmpflags(words/8),tmpdata(words*4),flags(words/8),data(words*4);
  input.seekg(0);input.read(reinterpret_cast<char*>(in.host.data()),bytes);
  int minimum=1;
  auto begin=std::chrono::steady_clock::now();
  launch("compress_match",size_t(blocks)*128,128,
    {&in.mem,&blocks,&flagsizes.mem,&sizes.mem,&tmpflags.mem,&tmpdata.mem,&minimum},
    {iris_r,sizeof(blocks),iris_rw,iris_rw,iris_rw,iris_rw,sizeof(minimum)});
  launch("exclusive_scan",1,1,{&flagsizes.mem,&foff.mem,&count},{iris_r,iris_w,sizeof(count)});
  launch("exclusive_scan",1,1,{&sizes.mem,&offsets.mem,&count},{iris_r,iris_w,sizeof(count)});
  launch("compress_pack",size_t(blocks)*128,128,
    {&blocks,&foff.mem,&offsets.mem,&tmpflags.mem,&tmpdata.mem,&flags.mem,&data.mem},
    {sizeof(blocks),iris_r,iris_r,iris_r,iris_r,iris_w,iris_w},
    {&foff.mem,&offsets.mem,&flags.mem,&data.mem});
  auto compressed=std::chrono::steady_clock::now();
  launch("decompress",((size_t(blocks)+31)/32)*32,32,
    {&out.mem,&blocks,&foff.mem,&offsets.mem,&flags.mem,&data.mem},
    {iris_rw,sizeof(blocks),iris_r,iris_r,iris_r,iris_r},{&out.mem});
  auto end=std::chrono::steady_clock::now();
  bool ok=std::memcmp(in.host.data(),out.host.data(),bytes)==0;
  std::printf("Whole-output lossless roundtrip: %u bytes %s\n",bytes,ok?"PASS":"FAIL");
  if(const char* path=std::getenv("HECBENCH_DUMP")) {
    FILE* f=std::fopen(path,"wb");if(!f)return 1;
    std::fwrite(foff.host.data(),4,count,f);std::fwrite(offsets.host.data(),4,count,f);
    std::fwrite(flags.host.data(),1,foff.host.back(),f);
    std::fwrite(data.host.data(),1,offsets.host.back(),f);
    std::fwrite(out.host.data(),1,bytes,f);std::fclose(f);
  }
  float compressedsize=8*count+foff.host.back()+offsets.host.back();
  float ctime=std::chrono::duration<float,std::milli>(compressed-begin).count();
  float dtime=std::chrono::duration<float,std::milli>(end-compressed).count();
  std::printf("compression ratio: %g\ncompression e2e throughput: %g GB/s\ndecompression e2e throughput: %g GB/s\n",
    bytes/compressedsize,float(bytes)/1024/1024/ctime,float(bytes)/1024/1024/dtime);
  platform.finalize();return ok?0:1;
}
