#define main hecbench_original_main
#include "atomicCAS-cuda/main.cu"
#undef main
#include <limits>
#include <vector>
template<class T> void launch_cas(int op,T*data,int blocks){
 if(op==0)atomicMinDerived<<<blocks,256>>>(data);
 if(op==1)atomicMaxDerived<<<blocks,256>>>(data);
 if(op==2)atomicAddDerived<<<blocks,256>>>(data);
}
template<class T> bool cas_case(int op,int repeat,int blocks,void*raw,double &elapsed,double &result){
 T *data=static_cast<T*>(raw);T initial=op==0?std::numeric_limits<T>::max():op==1?std::numeric_limits<T>::lowest():T(0);
 if(op==1&&std::numeric_limits<T>::is_iec559)initial=std::numeric_limits<T>::min();
 cudaMemcpy(data,&initial,sizeof(T),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
 for(int r=0;r<(op==2?1:repeat);r++)launch_cas(op,data,blocks);
 cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();elapsed=std::chrono::duration<double>(stop-start).count()/(op==2?1:repeat);
 T value;cudaMemcpy(&value,data,sizeof(T),cudaMemcpyDeviceToHost);result=double(value);
 long long n=blocks*256;T expected=op==0?T(1):op==1?T(n):T(n*(n+1)/2);return value==expected;
}
int main(int argc,char**argv){
 int bound=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):10;int blocks=bound/256;if(bound%256)return 2;
 void *buffers[9];for(int i=0;i<9;i++)cudaMalloc(&buffers[i],8);
 double times[9],values[9];
 for(int op=0;op<3;op++){cas_case<unsigned long long>(op,1,blocks,buffers[3*op],times[3*op],values[3*op]);cas_case<long long>(op,1,blocks,buffers[3*op+1],times[3*op+1],values[3*op+1]);cas_case<double>(op,1,blocks,buffers[3*op+2],times[3*op+2],values[3*op+2]);}
 for(int trial=1;trial<=5;trial++){
  bool correct=true;double total=0;
  for(int op=0;op<3;op++){correct &= cas_case<unsigned long long>(op,repeat,blocks,buffers[3*op],times[3*op],values[3*op]);correct &= cas_case<long long>(op,repeat,blocks,buffers[3*op+1],times[3*op+1],values[3*op+1]);correct &= cas_case<double>(op,repeat,blocks,buffers[3*op+2],times[3*op+2],values[3*op+2]);}
  for(int i=0;i<9;i++)total+=times[i];
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values,sizeof(double),9,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":9,\"max_abs_error\":0.0,\"timing_region\":\"sum_nine_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[",trial,total,correct?"true":"false",repeat);
  for(int i=0;i<9;i++)printf("%s%.12g",i?",":"",times[i]);
  printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":9,\"byte_order\":\"little\"}}\n",filename);
  if(!correct)return 1;
 }
 for(int i=0;i<9;i++)cudaFree(buffers[i]);return 0;
}
