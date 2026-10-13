#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char** argv) {
  if(argc!=2){printf("Usage: %s <repeat>\n",argv[0]);return 1;}
  int repeat=atoi(argv[1]);if(repeat<1)return 1;
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS)return 1;
  bool gpu=false;for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }if(!gpu)return 1;
  unsigned n=128*128*128*2,cta=128,wg=256,nbits=4,blocks=n/2/cta,arrayLength=blocks*16,size=4*wg,top=arrayLength/size;
  std::vector<unsigned> keys(n),temp(n),counters(arrayLength),offsets(arrayLength),counterSum(arrayLength),buffer(top);
  srand(95123);for(unsigned i=0;i<n;i++)keys[i]=((rand()&0xffff)<<16)|(rand()&0xffff);
  std::vector<unsigned> ref(keys);std::sort(ref.begin(),ref.end());
  iris::DMem mk(keys.data(),n*sizeof(unsigned)),mt(temp.data(),n*sizeof(unsigned)),mc(counters.data(),arrayLength*sizeof(unsigned)),mo(offsets.data(),arrayLength*sizeof(unsigned)),ms(counterSum.data(),arrayLength*sizeof(unsigned)),mb(buffer.data(),top*sizeof(unsigned));
  auto launch=[&](const char* name,size_t groups,size_t threads,std::vector<void*> args,std::vector<int> info,bool flush=false){
    iris::Task task;size_t gws[]={groups*threads},lws[]={threads};
    if(task.kernel(name,1,nullptr,gws,lws,args.size(),args.data(),info.data())!=IRIS_SUCCESS)exit(2);
    if(flush)task.flush_out(mk);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(2);
  };
  auto start=std::chrono::steady_clock::now();
  for(int r=0;r<repeat;r++)for(unsigned bit=0;bit<32;bit+=4) {
    launch("radixSortBlocksKeysK",n/4/cta,cta,{&mk,&mt,&nbits,&bit},{iris_r,iris_w,sizeof(nbits),sizeof(bit)});
    launch("findRadixOffsetsK",blocks,cta,{&mt,&mc,&mo,&bit,&blocks},{iris_r,iris_w,iris_w,sizeof(bit),sizeof(blocks)});
    launch("scanExclusiveLocal1K",arrayLength/4/wg,wg,{&ms,&mc,&size},{iris_w,iris_r,sizeof(size)});
    launch("scanExclusiveLocal2K",(top+wg-1)/wg,wg,{&mb,&ms,&mc,&top,&top},{iris_w,iris_r,iris_r,sizeof(top),sizeof(top)});
    launch("uniformUpdateK",top,wg,{&ms,&mb},{iris_rw,iris_r});
    launch("reorderDataKeysOnlyK",blocks,cta,{&mk,&mt,&mo,&ms,&bit,&n,&blocks},{iris_rw,iris_r,iris_r,iris_r,sizeof(bit),sizeof(n),sizeof(blocks)},r==repeat-1 && bit==28);
  }
  printf("Average execution time of radixsort: %f (s)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat);
  bool ok=keys==ref;printf("%s\n",ok ? "PASS" : "FAIL");printf("FULL_CHECK keys=%u multiset_and_order %s\n",n,ok ? "PASS" : "FAIL");
  if(const char* path=getenv("HECBENCH_ARTIFACT")){FILE* f=fopen(path,"wb");if(!f)return 3;fwrite(keys.data(),sizeof(unsigned),n,f);fclose(f);}
  int errors=platform.error_count();platform.finalize();return ok && !errors ? 0 : 1;
}
