#define main hecbench_original_main
#include "atomicCost-cuda/main.cu"
#undef main
#include <vector>
template<class T>bool run_cost(int phase,void*raw,int n,int size,int repeat,double &time,std::vector<double>&values){
 T*data=static_cast<T*>(raw);cudaMemset(data,0,n*sizeof(T));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
 for(int r=0;r<repeat;r++){if(phase==0)wiAtomicOnGlobalMem<<<n/256,256>>>(data,size);else woAtomicOnGlobalMem<<<n/256,256>>>(data,size);}
 cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();time=std::chrono::duration<double>(stop-start).count()/repeat;
 std::vector<T>result(n);cudaMemcpy(result.data(),data,n*sizeof(T),cudaMemcpyDeviceToHost);bool correct=true;
 for(int j=0;j<n;j++){int sum=0;for(int i=j*size;i<(j+1)*size;i++)sum+=i%2;correct &= result[j]==T(sum*repeat);values.push_back(double(result[j]));}return correct;
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,size=argc>2?atoi(argv[2]):4,repeat=argc>3?atoi(argv[3]):10;if(n%256||size<1||size>16)return 2;
 void*buffers[6];for(int i=0;i<6;i++)cudaMalloc(&buffers[i],n*8);double times[6];std::vector<double>warm;
 for(int phase=0;phase<2;phase++){run_cost<double>(phase,buffers[phase],n,size,1,times[phase],warm);run_cost<int>(phase,buffers[2+phase],n,size,1,times[2+phase],warm);run_cost<float>(phase,buffers[4+phase],n,size,1,times[4+phase],warm);}
 for(int trial=1;trial<=5;trial++){
  bool correct=true;std::vector<double>values;double total=0;
  for(int phase=0;phase<2;phase++)correct &= run_cost<double>(phase,buffers[phase],n,size,repeat,times[phase],values);
  for(int phase=0;phase<2;phase++)correct &= run_cost<int>(phase,buffers[2+phase],n,size,repeat,times[2+phase],values);
  for(int phase=0;phase<2;phase++)correct &= run_cost<float>(phase,buffers[4+phase],n,size,repeat,times[4+phase],values);
  for(int i=0;i<6;i++)total+=times[i];char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(double),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"sum_six_atomic_vs_nonatomic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[",trial,total,correct?"true":"false",6*n,repeat);
  for(int i=0;i<6;i++)printf("%s%.12g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",filename,6*n);if(!correct)return 1;
 }for(int i=0;i<6;i++)cudaFree(buffers[i]);return 0;
}
