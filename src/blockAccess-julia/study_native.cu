#define main hecbench_original_main
#include "blockAccess-cuda/main.cu"
#undef main
#include <vector>
#include <cstdint>
void launch_blockaccess(int phase,float*input,unsigned char*output,int n,int blocks){
 if(phase==0)reference<<<blocks,256>>>(input,output,n);else kernel<256,4><<<blocks,256>>>(input,output,n);
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):65536,repeat=argc>2?atoi(argv[2]):10;if(n%4)return 2;int blocks=(n+1023)/1024;
 std::vector<float>input(n);for(int i=0;i<n;i++)input[i]=float((i*37)%1024-512)/4.f;float*dx;unsigned char*dy;cudaMalloc(&dx,n*sizeof(float));cudaMalloc(&dy,n);cudaMemcpy(dx,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);for(int p=0;p<2;p++)launch_blockaccess(p,dx,dy,n,blocks);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[2];bool correct=true;std::vector<uint32_t>values;std::vector<unsigned char>result(n);
  for(int phase=0;phase<2;phase++){
   cudaMemset(dy,0,n);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int r=0;r<repeat;r++)launch_blockaccess(phase,dx,dy,n,blocks);cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[phase]=std::chrono::duration<double>(stop-start).count()/repeat;cudaMemcpy(result.data(),dy,n,cudaMemcpyDeviceToHost);
   for(int i=0;i<n;i++){correct &= result[i]==static_cast<unsigned char>(int(input[i]));values.push_back(uint32_t(result[i]));}
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(uint32_t),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_float_to_byte_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],correct?"true":"false",2*n,repeat,times[0],times[1],filename,2*n);if(!correct)return 1;
 }cudaFree(dx);cudaFree(dy);return 0;
}
