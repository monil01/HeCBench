#define main original_hecbench_main
#include "../zeropoint-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=262144,repeats=10;std::vector<float>lo(n),hi(n),sc(n),refsc(n);std::vector<int>zp(n),refzp(n);std::vector<double>got(2*n),expected(2*n);
for(int i=0;i<n;i++){lo[i]=(float)((17*i)%257-128)/128.f;hi[i]=(float)((13*i)%257-128)/128.f;}reference(lo.data(),hi.data(),-127,127,n,true,refsc.data(),refzp.data());for(int i=0;i<n;i++){expected[i]=refsc[i];expected[n+i]=refzp[i];}
float *dl,*dh,*ds;int*dz;if(cudaMalloc(&dl,n*4)!=cudaSuccess||cudaMalloc(&dh,n*4)!=cudaSuccess||cudaMalloc(&ds,n*4)!=cudaSuccess||cudaMalloc(&dz,n*4)!=cudaSuccess)return 3;cudaMemcpy(dl,lo.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dh,hi.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){zero_point<<<n/256,256>>>(dl,dh,-127,127,n,true,ds,dz);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(ds,0,n*4);cudaMemset(dz,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(sc.data(),ds,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(zp.data(),dz,n*4,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++){got[i]=sc[i];got[n+i]=zp[i];}
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(double),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
