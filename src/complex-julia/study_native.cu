#define main hecbench_original_main
#include "complex-cuda/main.cu"
#undef main
#include <vector>
#include <cstdint>
void launch_complex(int phase,char*data,int n){
 if(phase==0)complex_float<<<(n+255)/256,256>>>(data,n);
 if(phase==1)ref_complex_float<<<(n+255)/256,256>>>(data,n);
 if(phase==2)complex_double<<<(n+255)/256,256>>>(data,n);
 if(phase==3)ref_complex_double<<<(n+255)/256,256>>>(data,n);
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):10;char*data;cudaMalloc(&data,n);for(int p=0;p<4;p++)launch_complex(p,data,n);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[4],total=0;bool correct=true;std::vector<uint32_t>values;std::vector<char>result(n);
  for(int phase=0;phase<4;phase++){
   cudaMemset(data,0,n);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int r=0;r<repeat;r++)launch_complex(phase,data,n);cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[phase]=std::chrono::duration<double>(stop-start).count()/repeat;total+=times[phase];cudaMemcpy(result.data(),data,n,cudaMemcpyDeviceToHost);
   for(char v:result){correct &= v==5;values.push_back(uint32_t(v));}
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(uint32_t),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"sum_four_complex_property_score_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,total,correct?"true":"false",4*n,repeat,times[0],times[1],times[2],times[3],filename,4*n);if(!correct)return 1;
 }cudaFree(data);return 0;
}
