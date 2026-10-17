#define main original_hecbench_main
#include "../boxfilter-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int w=256,h=128,repeats=10,n=w*h;std::vector<unsigned int> input(n),cpu(n,0),ctemp(n,0),packed(n);std::vector<int64_t> got(n),expected(n);
for(int y=0;y<h;y++)for(int x=0;x<w;x++)input[y*w+x]=((13*x+7*y)%256)|(((3*x+11*y)%256)<<8)|(((17*x+5*y)%256)<<16)|(((19*x+23*y)%256)<<24);
BoxFilterHost(input.data(),ctemp.data(),cpu.data(),w,h,10,1.f/21.f);for(int i=0;i<n;i++)expected[i]=i<w?0:(int64_t)cpu[i];
uchar4*in;unsigned int *temp,*out;cudaMalloc(&in,n*4);cudaMalloc(&temp,n*4);cudaMalloc(&out,n*4);cudaMemcpy(in,input.data(),n*4,cudaMemcpyHostToDevice);

 auto launch=[&](){row_kernel<<<dim3(4,h),90,360>>>(in,temp,w,h,10,16,1.f/21.f,64);col_kernel<<<4,64>>>(temp,out,w,h,10,1.f/21.f);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(temp,0,n*4);cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(packed.data(),out,n*4,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++)got[i]=(int64_t)packed[i];
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int64_t),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
