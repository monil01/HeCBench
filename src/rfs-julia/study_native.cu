#define main hecbench_original_main
#include "rfs-cuda/main.cu"
#undef main
#include <vector>
#include <cstdint>
void run_rfs(int phase,float*input,float*output,float*maxval,int arrays,int len,float factor){
 if(phase==0){for(int a=0;a<arrays;a++)sumArray<<<256,256>>>(factor,len,input+a*len,output+a);}
 else sumArrays<<<256,256>>>(arrays,len,input,output,maxval);
}
int main(int argc,char**argv){
 int arrays=argc>1?atoi(argv[1]):16,len=argc>2?atoi(argv[2]):512,n=arrays*len;std::vector<float>input(n),expected(arrays,0.f),maxval(arrays,1.f),result(arrays);float factor=createRoundingFactor(1.f,len);
 for(int i=0;i<n;i++){float raw=float((int64_t(i)*1103515245+12345)&0x00ffffff)/16777216.f;input[i]=(i%len==0)?1.f:((i%2==0)?raw:-raw);}
 for(int a=0;a<arrays;a++)for(int i=0;i<len;i++)expected[a]+=truncateWithRoundingFactor(factor,input[a*len+i]);
 float *dx,*dy,*dm;cudaMalloc(&dx,n*sizeof(float));cudaMalloc(&dy,arrays*sizeof(float));cudaMalloc(&dm,arrays*sizeof(float));cudaMemcpy(dx,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dm,maxval.data(),arrays*sizeof(float),cudaMemcpyHostToDevice);for(int p=0;p<2;p++){cudaMemset(dy,0,arrays*sizeof(float));run_rfs(p,dx,dy,dm,arrays,len,factor);}cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  double times[2];bool correct=true;std::vector<float>values;
  for(int phase=0;phase<2;phase++){
   cudaMemset(dy,0,arrays*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();run_rfs(phase,dx,dy,dm,arrays,len,factor);cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();times[phase]=std::chrono::duration<double>(stop-start).count();cudaMemcpy(result.data(),dy,arrays*sizeof(float),cudaMemcpyDeviceToHost);correct &= memcmp(result.data(),expected.data(),arrays*sizeof(float))==0;values.insert(values.end(),result.begin(),result.end());
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(float),values.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"sum_atomic_per_array_launches_and_reverse_single_launch\",\"jit_excluded\":true,\"internal_repeats\":1,\"subkernel_times_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],correct?"true":"false",2*arrays,times[0],times[1],filename,2*arrays);if(!correct)return 1;
 }cudaFree(dx);cudaFree(dy);cudaFree(dm);return 0;
}
