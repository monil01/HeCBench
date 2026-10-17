#define main original_hecbench_main
#include "../quant3MatMul-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int m=4096,n=1024,height=m/32*3,repeats=10;std::vector<float>vec(m),scales(n),zeros(n),got(n),expected(n,0.f);std::vector<unsigned int>packed(height*n,0);for(int r=0;r<m;r++)vec[r]=(float)(r%17-8)/8.f;for(int c=0;c<n;c++){scales[c]=(float)(c%7+1)/8.f;zeros[c]=(float)(c%5-2)/4.f;}for(int r=0;r<m;r++)for(int c=0;c<n;c++){unsigned q=(17*r+13*c)%8;int bit=3*r,word=bit/32,shift=bit%32;packed[word*n+c]|=q<<shift;if(shift>29)packed[(word+1)*n+c]|=q>>(32-shift);expected[c]+=(scales[c]*(float)q-zeros[c])*vec[r];}float *dv,*ds,*dz,*out;int*dm;if(cudaMalloc(&dv,m*4)!=cudaSuccess||cudaMalloc(&ds,n*4)!=cudaSuccess||cudaMalloc(&dz,n*4)!=cudaSuccess||cudaMalloc(&out,n*4)!=cudaSuccess||cudaMalloc(&dm,height*n*4)!=cudaSuccess)return 3;cudaMemcpy(dv,vec.data(),m*4,cudaMemcpyHostToDevice);cudaMemcpy(ds,scales.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dz,zeros.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dm,packed.data(),height*n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){cudaMemset(out,0,n*4);VecQuant3MatMulKernel<float><<<dim3(16,4),256>>>(dv,dm,out,ds,dz,height,n);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
