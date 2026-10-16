#define main hecbench_original_main
#include "unfold-cuda/main.cu"
#undef main
int main(int argc,char **argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):100;
 int *input=new int[n],*output=new int[n];for(int i=0;i<n;i++)input[i]=i%256;
 int *dx,*dy;int64_t *idx;int64_t zero=0;cudaMalloc(&dx,n*sizeof(int));cudaMalloc(&dy,n*sizeof(int));cudaMalloc(&idx,sizeof(int64_t));
 cudaMemcpy(dx,input,n*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(idx,&zero,sizeof(zero),cudaMemcpyHostToDevice);cudaMemset(dy,0,n*sizeof(int));
 unfold_backward_internal_kernel(dy,dx,idx,2,1,1,1,n,1);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  cudaMemset(dy,0,n*sizeof(int));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int r=0;r<repeat;r++)unfold_backward_internal_kernel(dy,dx,idx,2,1,1,1,n,1);
  cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();double seconds=std::chrono::duration<double>(stop-start).count()/repeat;
  cudaMemcpy(output,dy,n*sizeof(int),cudaMemcpyDeviceToHost);bool correct=true;for(int i=0;i<n;i++)correct &= output[i]==repeat*input[i];
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(output,sizeof(int),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,correct?"true":"false",n,repeat,filename,n);
  if(!correct)return 1;
 }
 cudaFree(dx);cudaFree(dy);cudaFree(idx);delete[]input;delete[]output;
}
