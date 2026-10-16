#define main hecbench_original_main
#include "atomicAggregate-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int blocks=argc>1?atoi(argv[1]):64,repeat=argc>2?atoi(argv[2]):10;
 int sizes[6]={32,16,8,4,2,1};int*buffers[6];for(int j=0;j<6;j++){cudaMalloc(&buffers[j],sizes[j]*sizeof(int));cudaMemset(buffers[j],0,sizes[j]*sizeof(int));k<<<blocks,256>>>(buffers[j],sizes[j]);}cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[6],total=0;std::vector<int>values;bool correct=true;
  for(int j=0;j<6;j++){
   int s=sizes[j];cudaMemset(buffers[j],0,s*sizeof(int));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
   for(int r=0;r<repeat;r++)k<<<blocks,256>>>(buffers[j],s);
   cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[j]=std::chrono::duration<double>(stop-start).count()/repeat;total+=times[j];
   std::vector<int>result(s);cudaMemcpy(result.data(),buffers[j],s*sizeof(int),cudaMemcpyDeviceToHost);for(int v:result){correct &= v==(256/s)*blocks*repeat;values.push_back(v);}
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(int),63,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":63,\"max_abs_error\":0.0,\"timing_region\":\"sum_six_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[",trial,total,correct?"true":"false",repeat);
  for(int i=0;i<6;i++)printf("%s%.12g",i?",":"",times[i]);
  printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":63,\"byte_order\":\"little\"}}\n",filename);if(!correct)return 1;
 }for(int i=0;i<6;i++)cudaFree(buffers[i]);return 0;
}
