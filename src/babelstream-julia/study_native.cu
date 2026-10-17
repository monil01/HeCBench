#define main original_hecbench_main
#include "../babelstream-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=262144,repeats=10;std::string op=argc>1?argv[1]:"copy";
std::vector<float> a(n),b(n),c(n);for(int i=0;i<n;i++){a[i]=(i%31-15)/32.f;b[i]=((7*i)%17-8)/16.f;c[i]=((11*i)%23-11)/16.f;}
int count=op=="dot"?256:n;std::vector<float> got(count),expected(count);
if(op=="dot"){for(int group=0;group<256;group++){double s=0;for(int lane=0;lane<256;lane++)for(int i=group*256+lane;i<n;i+=65536)s+=(double)a[i]*b[i];expected[group]=(float)s;}}
else for(int i=0;i<n;i++){if(op=="copy")expected[i]=a[i];if(op=="mul")expected[i]=0.4f*c[i];if(op=="add")expected[i]=a[i]+b[i];if(op=="triad")expected[i]=b[i]+0.4f*c[i];if(op=="nstream"){expected[i]=a[i];for(int j=0;j<repeats;j++)expected[i]+=b[i]+0.4f*c[i];}}
float *da,*db,*dc,*partial;cudaMalloc(&da,n*4);cudaMalloc(&db,n*4);cudaMalloc(&dc,n*4);cudaMalloc(&partial,256*4);
auto reset=[&](){cudaMemcpy(da,a.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dc,c.data(),n*4,cudaMemcpyHostToDevice);cudaMemset(partial,0,1024);};reset();

 auto launch=[&](){if(op=="copy")copy_kernel<<<n/256,256>>>(da,dc);else if(op=="mul")mul_kernel<<<n/256,256>>>(db,dc);else if(op=="add")add_kernel<<<n/256,256>>>(da,db,dc);else if(op=="triad")triad_kernel<<<n/256,256>>>(da,db,dc);else if(op=="nstream")nstream_kernel<<<n/256,256>>>(da,db,dc);else if(op=="dot")dot_kernel<<<256,256>>>(da,db,partial,n);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 reset();cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 float* output=(op=="copy"||op=="add")?dc:op=="mul"?db:op=="dot"?partial:da;cudaMemcpy(got.data(),output,count*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.0001));ok&=std::isfinite((double)got[j])&&e<=0.0001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
