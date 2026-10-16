// Named study fixture: sequential Int32 input, same as the native default.
#define main hecbench_original_main
#include "../stencil1d-cuda/stencil_1d.cu"
#undef main
#include <vector>
#include <cstdint>
#include <unistd.h>
int main(int argc,char**argv) {
  int n=argc>1?atoi(argv[1]):1048576, repeat=argc>2?atoi(argv[2]):100;
  if(n<=0 || n%256 || repeat<=0) return 2;
  std::vector<int> input(n+RADIUS), output(n), expected(n);
  for(int i=0;i<n+RADIUS;i++)input[i]=i;
  for(int i=0;i<n;i++){int s=0;for(int o=-RADIUS;o<=RADIUS;o++)if(i+o>=0)s+=input[i+o];expected[i]=s;}
  int *a,*b;cudaMalloc(&a,(n+RADIUS)*sizeof(int));cudaMalloc(&b,n*sizeof(int));
  cudaMemcpy(a,input.data(),(n+RADIUS)*sizeof(int),cudaMemcpyHostToDevice);
  stencil_1d<<<n/256,256>>>(a,b);cudaDeviceSynchronize();
  bool allok=true;
  for(int t=1;t<=5;t++) {
    cudaMemset(b,0,n*sizeof(int));cudaDeviceSynchronize();
    auto start=std::chrono::steady_clock::now();
    for(int j=0;j<repeat;j++)stencil_1d<<<n/256,256>>>(a,b);
    cudaError_t status=cudaDeviceSynchronize();
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;
    cudaMemcpy(output.data(),b,n*sizeof(int),cudaMemcpyDeviceToHost);
    long long maxabs=0;for(int i=0;i<n;i++){long long d=std::llabs((long long)output[i]-expected[i]);if(d>maxabs)maxabs=d;}
    bool ok=status==cudaSuccess && maxabs==0;allok &= ok;
    char artifact[]="/tmp/hecbench-stencil1d-XXXXXX";int fd=mkstemp(artifact);FILE *af=fdopen(fd,"wb");fwrite(output.data(),sizeof(int),n,af);fclose(af);
    printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%lld,\"max_rel_error\":0.0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n,maxabs,repeat,artifact,n);
  }
  cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
