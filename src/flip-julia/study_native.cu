#define main hecbench_original_main
#include "flip-cuda/main.cu"
#undef main
int main(int argc,char **argv){
 int d=argc>1?atoi(argv[1]):16,repeat=argc>2?atoi(argv[2]):100;int64_t n=int64_t(d)*d*d;
 float *input=new float[n],*output=new float[n];for(int64_t i=0;i<n;i++)input[i]=float(i);
 float *dx,*dy;int64_t *dims,*strides,*shape;int64_t hdims[3]={0,1,2},hstrides[3]={int64_t(d)*d,d,1},hshape[3]={d,d,d};
 cudaMalloc(&dx,n*sizeof(float));cudaMalloc(&dy,n*sizeof(float));cudaMalloc(&dims,3*sizeof(int64_t));cudaMalloc(&strides,3*sizeof(int64_t));cudaMalloc(&shape,3*sizeof(int64_t));
 cudaMemcpy(dx,input,n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dims,hdims,sizeof(hdims),cudaMemcpyHostToDevice);cudaMemcpy(strides,hstrides,sizeof(hstrides),cudaMemcpyHostToDevice);cudaMemcpy(shape,hshape,sizeof(hshape),cudaMemcpyHostToDevice);
 flip_kernel<<<(n+255)/256,256>>>(dx,dy,n,dims,3,strides,strides,shape,3);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  cudaMemset(dy,0,n*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int r=0;r<repeat;r++)flip_kernel<<<(n+255)/256,256>>>(dx,dy,n,dims,3,strides,strides,shape,3);
  cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();double seconds=std::chrono::duration<double>(stop-start).count()/repeat;
  cudaMemcpy(output,dy,n*sizeof(float),cudaMemcpyDeviceToHost);bool correct=true;for(int64_t i=0;i<n;i++)correct &= output[i]==input[n-i-1];
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(output,sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%ld,\"max_abs_error\":0.0,\"timing_region\":\"synchronized_launch_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%ld,\"byte_order\":\"little\"}}\n",trial,seconds,correct?"true":"false",n,repeat,filename,n);
  if(!correct)return 1;
 }
 cudaFree(dx);cudaFree(dy);cudaFree(dims);cudaFree(strides);cudaFree(shape);delete[]input;delete[]output;
}
