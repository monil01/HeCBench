// Measurement adapter invokes unchanged native CUDA device kernel.
#define main hecbench_original_main
#include "saxpy-ompt-cuda/main.cu"
#undef main
int main(int argc,char **argv) {
  int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):100;
  float a=2.0f; float *x=new float[n],*y=new float[n],*result=new float[n];
  for(int i=0;i<n;i++) {
    x[i]=1.0f+float((i*2654435761u)&0xffff)/65535.0f;
    y[i]=0.5f+float((i*40503u)&0xffff)/65535.0f;
  }
  float *dx,*dy;cudaMalloc(&dx,n*sizeof(float));cudaMalloc(&dy,n*sizeof(float));
  cudaMemcpy(dx,x,n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dy,y,n*sizeof(float),cudaMemcpyHostToDevice);
  saxpy_k<<<(n+255)/256,256>>>(n,a,dx,dy);cudaDeviceSynchronize();
  for(int trial=1;trial<=5;trial++) {
    cudaMemcpy(dy,y,n*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();
    auto start=std::chrono::steady_clock::now();
    for(int r=0;r<repeat;r++) saxpy_k<<<(n+255)/256,256>>>(n,a,dx,dy);
    cudaDeviceSynchronize(); auto stop=std::chrono::steady_clock::now();
    double seconds=std::chrono::duration<double>(stop-start).count()/repeat;
    cudaMemcpy(result,dy,n*sizeof(float),cudaMemcpyDeviceToHost);
    float error=0;bool correct=true;
    for(int i=0;i<n;i++){float d=fabsf(result[i]-(y[i]+float(repeat)*a*x[i]));error=fmaxf(error,d);correct &= std::isfinite(result[i])&&d<1e-2f;}
    char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);
    FILE *f=fopen(filename,"wb");fwrite(result,sizeof(float),n,f);fclose(f);
    printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,correct?"true":"false",n,error,repeat,filename,n);
    if(!correct)return 1;
  }
  cudaFree(dx);cudaFree(dy);delete[] x;delete[] y;delete[] result;
}
