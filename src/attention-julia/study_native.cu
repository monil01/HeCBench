#define main original_hecbench_main
#include "../attention-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=1024,d=64,repeats=10;std::vector<float>key(n*d),value(n*d),query(d),rd(n),rs(n),ro(d,0.f),got(2*n+1+d),expected(2*n+1+d);for(int i=0;i<n*d;i++){key[i]=(float)((17*i)%31-15)/128.f;value[i]=(float)((19*i)%127-63)/64.f;}for(int j=0;j<d;j++)query[j]=(float)((13*j)%17-8)/64.f;float re=0;for(int i=0;i<n;i++){float a=0;for(int j=0;j<d;j++)a+=key[i*d+j]*query[j];rd[i]=a;re+=expf(a);}for(int i=0;i<n;i++)rs[i]=expf(rd[i])/re;for(int j=0;j<d;j++)for(int i=0;i<n;i++)ro[j]+=rs[i]*value[i*d+j];std::copy(rd.begin(),rd.end(),expected.begin());expected[n]=re;std::copy(rs.begin(),rs.end(),expected.begin()+n+1);std::copy(ro.begin(),ro.end(),expected.begin()+2*n+1);float *dk,*dv,*dq,*dot,*es,*score,*out;if(cudaMalloc(&dk,n*d*4)!=cudaSuccess||cudaMalloc(&dv,n*d*4)!=cudaSuccess||cudaMalloc(&dq,d*4)!=cudaSuccess||cudaMalloc(&dot,n*4)!=cudaSuccess||cudaMalloc(&es,4)!=cudaSuccess||cudaMalloc(&score,n*4)!=cudaSuccess||cudaMalloc(&out,d*4)!=cudaSuccess)return 3;cudaMemcpy(dk,key.data(),n*d*4,cudaMemcpyHostToDevice);cudaMemcpy(dv,value.data(),n*d*4,cudaMemcpyHostToDevice);cudaMemcpy(dq,query.data(),d*4,cudaMemcpyHostToDevice);
 auto launch=[&](){cudaMemset(es,0,4);attention_kernel1<<<4,256>>>(dk,dq,dot,es,n,d);attention_kernel2<<<4,256>>>(es,dot,score,n);attention_kernel3<<<1,256>>>(score,dv,out,n,d);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(dot,0,n*4);cudaMemset(es,0,4);cudaMemset(score,0,n*4);cudaMemset(out,0,d*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),dot,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+n,es,4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+n+1,score,n*4,cudaMemcpyDeviceToHost);cudaMemcpy(got.data()+2*n+1,out,d*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001+1e-6*fabs((double)expected[j]);}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
