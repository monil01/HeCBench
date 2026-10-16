#define main hecbench_original_main
#include "atomicPerf-cuda/main.cu"
#undef main
#include <vector>
template<class T>void launch_perf(int phase,T*data,int n,int blocks,int offset){
 if(phase==0)BlockRangeAtomicOnGlobalMem<<<blocks,256>>>(data,n);
 if(phase==1)WarpRangeAtomicOnGlobalMem<<<blocks,256>>>(data,n);
 if(phase==2)SingleRangeAtomicOnGlobalMem<<<blocks,256>>>(data,offset,n);
}
template<class T>bool run_perf(int phase,void*raw,int n,int repeat,double &time,std::vector<double>&values){
 T*data=static_cast<T*>(raw);std::vector<T>initial(1024),result(1024),expected(1024);for(int i=0;i<1024;i++)initial[i]=T(i+1);expected=initial;
 cudaMemcpy(data,initial.data(),1024*sizeof(T),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
 for(int r=0;r<repeat;r++)launch_perf(phase,data,n,n/256,r%256);
 cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();time=std::chrono::duration<double>(stop-start).count()/repeat;
 cudaMemcpy(result.data(),data,1024*sizeof(T),cudaMemcpyDeviceToHost);bool correct=true;
 for(int r=0;r<repeat;r++)for(int i=0;i<n;i++){int index=phase==0?i%256:phase==1?i%32:r%256;expected[index]+=T(1);}
 for(int i=0;i<1024;i++){correct &= result[i]==expected[i];values.push_back(double(result[i]));}return correct;
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):10;if(n%256)return 2;
 void*buffers[9];for(int i=0;i<9;i++)cudaMalloc(&buffers[i],1024*8);double times[9];std::vector<double>warm;
 for(int phase=0;phase<3;phase++){run_perf<double>(phase,buffers[phase],n,1,times[phase],warm);run_perf<int>(phase,buffers[3+phase],n,1,times[3+phase],warm);run_perf<float>(phase,buffers[6+phase],n,1,times[6+phase],warm);}
 for(int trial=1;trial<=5;trial++){
  bool correct=true;std::vector<double>values;double total=0;
  for(int phase=0;phase<3;phase++)correct &= run_perf<double>(phase,buffers[phase],n,repeat,times[phase],values);
  for(int phase=0;phase<3;phase++)correct &= run_perf<int>(phase,buffers[3+phase],n,repeat,times[3+phase],values);
  for(int phase=0;phase<3;phase++)correct &= run_perf<float>(phase,buffers[6+phase],n,repeat,times[6+phase],values);
  for(int i=0;i<9;i++)total+=times[i];char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(double),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":9216,\"max_abs_error\":0.0,\"timing_region\":\"sum_nine_global_atomic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[",trial,total,correct?"true":"false",repeat);
  for(int i=0;i<9;i++)printf("%s%.12g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":9216,\"byte_order\":\"little\"}}\n",filename);if(!correct)return 1;
 }for(int i=0;i<9;i++)cudaFree(buffers[i]);return 0;
}
