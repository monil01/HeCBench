#define main original_main
#include "../axpby-cuda/main.cu"
#undef main
#include <fstream>
int main(int argc,char**argv){
 int nt=argc>1?atoi(argv[1]):8,chunk=argc>2?atoi(argv[2]):4096,repeats=argc>3?atoi(argv[3]):10;
 if(nt<=0||repeats<=0||chunk<=0)return 2;
 std::vector<std::vector<Tensor<float>>> tl(3);
 std::vector<std::vector<float>> h(3),expected;std::vector<int> lens;
 for(int t=0;t<nt;t++){
   int len=1024+(((t+1)*7919)%16384);lens.push_back(len);expected.push_back(std::vector<float>(len));
   for(int d=0;d<3;d++){
     std::vector<float> v(len,0);if(d<2)for(int i=0;i<len;i++)v[i]=d==0?(t*17+i*3)%1009:(t*29+i*5)%1013;
     float *dev;cudaMalloc(&dev,len*sizeof(float));cudaMemcpy(dev,v.data(),len*sizeof(float),cudaMemcpyHostToDevice);
     Tensor<float> ten;ten.data_ptr=dev;ten.numel=len;tl[d].push_back(ten);
     if(d<2)for(int i=0;i<len;i++)expected[t][i]+=v[i];
   }
 }
 int *flag;cudaMalloc(&flag,sizeof(int));cudaMemset(flag,0,sizeof(int));Tensor<int> noop;noop.data_ptr=flag;noop.numel=1;
 auto launch=[&](){multi_tensor_apply<3>(BLOCK_SIZE,chunk,noop,tl,AxpbyFunctor<float,float,float>(),1.f,1.f,-1);};
 launch();cudaDeviceSynchronize();bool allok=true;
 for(int trial=1;trial<=5;trial++){
  for(int t=0;t<nt;t++)cudaMemset(tl[2][t].data_ptr,0,lens[t]*sizeof(float));cudaMemset(flag,0,sizeof(int));cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();for(int r=0;r<repeats;r++)launch();cudaDeviceSynchronize();
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  std::string file=std::string(getenv("STUDY_OUTPUT_DIR"))+"/axpby_"+std::to_string(trial)+".bin";std::ofstream out(file,std::ios::binary);
  bool ok=true;double maxerr=0;size_t count=0;
  for(int t=0;t<nt;t++){
    std::vector<float> v(lens[t]);cudaMemcpy(v.data(),tl[2][t].data_ptr,lens[t]*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<lens[t];i++){double e=fabs(v[i]-expected[t][i]);maxerr=fmax(maxerr,e);ok&=std::isfinite(v[i])&&e<=1e-3;}
    out.write((char*)v.data(),v.size()*sizeof(float));count+=v.size();
  }
  ok&=cudaGetLastError()==cudaSuccess;allok&=ok;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.9g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",count,maxerr,repeats,file.c_str(),count);
 }
 for(auto &list:tl)for(auto &t:list)cudaFree(t.data_ptr);cudaFree(flag);return allok?0:1;
}
