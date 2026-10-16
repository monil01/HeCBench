// Uses the unchanged native CUDA kernel; outer measurement protocol only.
#define main hecbench_original_main
#include "reverse-cuda/main.cu"
#undef main
int main(int argc, char **argv) {
  const int iterations = argc > 1 ? atoi(argv[1]) : 1;
  const int len = 256;
  int input[len], result[len];
  for (int i=0;i<len;i++) input[i]=i;
  int *device; cudaMalloc(&device, sizeof(input));
  cudaMemcpy(device,input,sizeof(input),cudaMemcpyHostToDevice);
  reverse<<<1,len>>>(device,len); cudaDeviceSynchronize();
  for (int trial=1;trial<=5;trial++) {
    std::default_random_engine generator(123);
    std::uniform_int_distribution<int> distribution(100,9999);
    double total=0; long launches=0; bool correct=true;
    for (int it=0;it<iterations;it++) {
      int count=distribution(generator); launches+=count;
      cudaMemcpy(device,input,sizeof(input),cudaMemcpyHostToDevice);
      cudaDeviceSynchronize(); auto start=std::chrono::steady_clock::now();
      for(int r=0;r<count;r++) reverse<<<1,len>>>(device,len);
      cudaDeviceSynchronize(); auto stop=std::chrono::steady_clock::now();
      total+=std::chrono::duration<double>(stop-start).count();
      cudaMemcpy(result,device,sizeof(result),cudaMemcpyDeviceToHost);
      for(int i=0;i<len;i++) correct &= result[i]==(count%2 ? len-i-1 : i);
    }
    char filename[64]; snprintf(filename,sizeof(filename),"trial_%d.bin",trial);
    FILE *artifact=fopen(filename,"wb"); fwrite(result,sizeof(int),len,artifact); fclose(artifact);
    printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"synchronized_kernel_launch_loops_total\",\"jit_excluded\":true,\"internal_repeats\":%ld,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":256,\"byte_order\":\"little\"}}\n",trial,total,correct?"true":"false",len*iterations,launches,filename);
    if(!correct) return 1;
  }
  cudaFree(device); return 0;
}
