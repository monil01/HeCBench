#define main original_hecbench_main
#include "../aidw-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=1024,repeats=10;const float area=4000000.f;
std::vector<float> dx(n),dy(n),dz(n),ix(n),iy(n),avg(n),expected(n),got(n);
for(int i=0;i<n;i++){dx[i]=16*(i%32)+1;dy[i]=16*(i/32)+2;dz[i]=11.f+(i%47)/8.f;ix[i]=16*(i%32)+7.25f;iy[i]=16*(i/32)+9.5f;avg[i]=2*(i%33);}
reference(dx.data(),dy.data(),dz.data(),n,ix.data(),iy.data(),expected.data(),n,area,avg.data());
float *ddx,*ddy,*ddz,*dix,*diy,*davg,*out;
cudaMalloc(&ddx,n*4);cudaMalloc(&ddy,n*4);cudaMalloc(&ddz,n*4);cudaMalloc(&dix,n*4);cudaMalloc(&diy,n*4);cudaMalloc(&davg,n*4);cudaMalloc(&out,n*4);
cudaMemcpy(ddx,dx.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(ddy,dy.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(ddz,dz.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dix,ix.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(diy,iy.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(davg,avg.data(),n*4,cudaMemcpyHostToDevice);

 auto launch=[&](){AIDW_Kernel<<<4,256>>>(ddx,ddy,ddz,n,dix,diy,out,n,area,davg);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
