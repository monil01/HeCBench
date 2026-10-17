#define main original_hecbench_main
#include "../attentionMultiHead-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int beams=4,heads=16,cols=4096,steps=9,repeats=10,n=beams*cols;const float scale=0.0625f;
std::vector<float> q(n),k(n*steps),v(n*steps),expected(n),got(n);
for(int i=0;i<n;i++)q[i]=(i%17-8)/16.f;
for(int i=0;i<n*steps;i++){k[i]=((7*i)%31-15)/32.f;v[i]=((11*i)%53-26)/32.f;}
mha_reference(q.data(),k.data(),v.data(),beams,steps,cols,cols,heads,scale,64,expected.data());
float *dq,*dk,*dv,*out;cudaMalloc(&dq,n*4);cudaMalloc(&dk,n*steps*4);cudaMalloc(&dv,n*steps*4);cudaMalloc(&out,n*4);cudaMemcpy(dq,q.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dk,k.data(),n*steps*4,cudaMemcpyHostToDevice);cudaMemcpy(dv,v.data(),n*steps*4,cudaMemcpyHostToDevice);

 auto launch=[&](){mha<<<64,256,1060>>>(dq,dk,dv,beams,steps,cols,cols,heads,scale,64,out);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
