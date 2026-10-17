#define main original_hecbench_main
#include "../adam-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=65536,steps=4,repeats=10;std::vector<float>hp(n),hm(n),hv(n),hg(n),rp(n),rm(n),rv(n),got(3*n),expected(3*n);for(int i=0;i<n;i++){hp[i]=(float)(i%29-14)/16.f;hm[i]=(float)(i%31+1)/64.f;hv[i]=(float)(i%17+1)/32.f;hg[i]=(float)(i%23-11)/8.f;}rp=hp;rm=hm;rv=hv;reference<float,float>(repeats,rp.data(),rm.data(),rv.data(),hg.data(),0.9f,0.999f,1e-8f,256.f,1e-3f,steps,n,ADAM_MODE_0,0.5f);std::copy(rp.begin(),rp.end(),expected.begin());std::copy(rm.begin(),rm.end(),expected.begin()+n);std::copy(rv.begin(),rv.end(),expected.begin()+2*n);float *dp,*dm,*dv,*dg;if(cudaMalloc(&dp,n*4)!=cudaSuccess||cudaMalloc(&dm,n*4)!=cudaSuccess||cudaMalloc(&dv,n*4)!=cudaSuccess||cudaMalloc(&dg,n*4)!=cudaSuccess)return 3;cudaMemcpy(dp,hp.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dm,hm.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dv,hv.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dg,hg.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){adam<float,float><<<256,256>>>(dp,dm,dv,dg,0.9f,0.999f,1e-8f,256.f,1e-3f,steps,n,ADAM_MODE_0,0.5f);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemcpy(dp,hp.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dm,hm.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dv,hv.data(),n*4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),dp,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+n,dm,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+2*n,dv,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
