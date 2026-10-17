#define main original_hecbench_main
#include "../lombscargle-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int ns=1000,nf=100000,repeats=10,blocks=(nf+255)/256*256;std::vector<float>host(2*ns+nf+1),got(nf),source_ref(nf);std::vector<double>expected(nf);FILE*a=fopen("/home/imo/HeCBench_julia/src/lombscargle-julia/study_input_f32.bin","rb");if(!a||fread(host.data(),4,host.size(),a)!=host.size())return 5;fclose(a);FILE*b=fopen("/home/imo/HeCBench_julia/src/lombscargle-julia/study_expected_f64.bin","rb");if(!b||fread(expected.data(),8,nf,b)!=nf)return 6;fclose(b);float*hx=host.data(),*hy=hx+ns,*hf=hy+ns;float scale=host.back();lombscargle_cpu(ns,nf,hx,hy,hf,source_ref.data(),scale);double energybound=0;for(int i=0;i<ns;i++)energybound+=0.5*(double)scale*hy[i]*hy[i];float*x,*y,*f,*out;cudaMalloc(&x,ns*4);cudaMalloc(&y,ns*4);cudaMalloc(&f,nf*4);cudaMalloc(&out,nf*4);cudaMemcpy(x,hx,ns*4,cudaMemcpyHostToDevice);cudaMemcpy(y,hy,ns*4,cudaMemcpyHostToDevice);cudaMemcpy(f,hf,nf*4,cudaMemcpyHostToDevice);
 auto launch=[&](){lombscargle<<<blocks,256>>>(ns,nf,x,y,f,out,scale);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,nf*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,nf*4,cudaMemcpyDeviceToHost);double source_error=0;int peak=0;for(int i=0;i<nf;i++){if(!std::isfinite(got[i])||got[i]<0)return 7;source_error=fmax(source_error,fabs((double)got[i]-source_ref[i]));if(got[i]>got[peak])peak=i;}if(source_error>0.1||fabs(hf[peak]-1.f)>0.002||got[peak]>energybound+0.1)return 8;printf("STUDY_SPECTRAL_CHECKS original_CPU_error=%.12g peak_angular_frequency=%.12g maximum_power=%.12g energy_bound=%.12g source_blocks=%d active_blocks=%d\n",source_error,hf[peak],got[peak],energybound,blocks,(nf+255)/256);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.1));ok&=std::isfinite((double)got[j])&&e<=0.1;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
