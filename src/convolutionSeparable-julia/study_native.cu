#define main original_hecbench_main
#include "../convolutionSeparable-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int w=512,h=256,n=w*h,repeats=10;std::vector<float>input(n),kernel(17),row(n),final(n),got(2*n),expected(2*n);for(int i=0;i<n;i++)input[i]=(float)((17*i+13*(i/w))%129-64)/64.f;for(int i=0;i<17;i++)kernel[i]=(float)((7*i)%17-8)/32.f;
for(int y=0;y<h;y++)for(int x=0;x<w;x++){double sum=0;for(int k=-8;k<=8;k++)if(x+k>=0&&x+k<w)sum+=(double)input[y*w+x+k]*kernel[8-k];row[y*w+x]=sum;}
for(int y=0;y<h;y++)for(int x=0;x<w;x++){double sum=0;for(int k=-8;k<=8;k++)if(y+k>=0&&y+k<h)sum+=(double)row[(y+k)*w+x]*kernel[8-k];final[y*w+x]=sum;}
for(int i=0;i<n;i++){expected[i]=row[i];expected[n+i]=final[i];}
float *di,*dk,*db,*out;cudaMalloc(&di,n*4);cudaMalloc(&dk,17*4);cudaMalloc(&db,n*4);cudaMalloc(&out,n*4);cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dk,kernel.data(),17*4,cudaMemcpyHostToDevice);

 auto launch=[&](){convolutionRows(db,di,dk,w,h,w);convolutionColumns(out,db,dk,w,h,w);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(db,0,n*4);cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),db,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+n,out,n*4,cudaMemcpyDeviceToHost);double delta=0,norm=0;for(int i=0;i<n;i++){double e=(double)got[n+i]-final[i];delta+=e*e;norm+=(double)final[i]*final[i];}if(!(sqrt(delta/norm)<1e-6))return 7;printf("STUDY_ACCURACY native_relative_L2=%.12g required_below=1e-6 pass=true\n",sqrt(delta/norm));
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-05));ok&=std::isfinite((double)got[j])&&e<=1e-05;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
