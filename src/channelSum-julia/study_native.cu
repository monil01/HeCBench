#define main original_hecbench_main
#include "../channelSum-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const bool nchw=argc>1&&std::string(argv[1])=="nchw";constexpr int n=4,c=128,hxw=1024,repeats=10,count=n*c*hxw;std::vector<int>input(count),got(2*c),expected(2*c,0),sums(c),squares(c);for(int ni=0;ni<n;ni++)for(int ci=0;ci<c;ci++)for(int hw=0;hw<hxw;hw++){int value=(17*(ni*hxw+hw)+13*ci)%19-9;int index=nchw?(ni*c+ci)*hxw+hw:(ni*hxw+hw)*c+ci;input[index]=value;expected[ci]+=value;expected[c+ci]+=value*value;}int*di,*ds,*dq;if(cudaMalloc(&di,count*4)!=cudaSuccess||cudaMalloc(&ds,c*4)!=cudaSuccess||cudaMalloc(&dq,c*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),count*4,cudaMemcpyHostToDevice);
 auto launch=[&](){if(nchw)ChannelSumNCHW<int,1,128><<<c,dim3(1,128)>>>(n,c,hxw,di,ds,dq);else ChannelSumNHWC<int><<<c,256>>>(n,c,hxw,di,ds,dq);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(ds,0,c*4);cudaMemset(dq,0,c*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),ds,c*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+c,dq,c*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
