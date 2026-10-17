#define main original_hecbench_main
#include "../keogh-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int N=262144,M=256,n=N-M+1,repeats=10;std::vector<float>data(N+2*n+3*M),source_ref(n),got(n);FILE*f=fopen("/home/imo/HeCBench_julia/src/keogh-julia/study_input_f32.bin","rb");if(!f||fread(data.data(),4,data.size(),f)!=data.size())return 5;fclose(f);float*s=data.data(),*av=s+N,*st=av+n,*lo=st+n,*up=lo+M;reference(s,av,st,source_ref.data(),lo,up,M,N);std::vector<double>expected(n);for(int start=0;start<n;start++){double residue=0;for(int i=0;i<M;i++){double v=((double)s[start+i]-av[start])/st[start];double distance=fmax(fmax((double)lo[i]-v,v-(double)up[i]),0.);residue+=distance*distance;}expected[start]=residue;}float *subject,*avgs,*stds,*lower,*upper,*out;cudaMalloc(&subject,N*4);cudaMalloc(&avgs,n*4);cudaMalloc(&stds,n*4);cudaMalloc(&lower,M*4);cudaMalloc(&upper,M*4);cudaMalloc(&out,n*4);cudaMemcpy(subject,s,N*4,cudaMemcpyHostToDevice);cudaMemcpy(avgs,av,n*4,cudaMemcpyHostToDevice);cudaMemcpy(stds,st,n*4,cudaMemcpyHostToDevice);cudaMemcpy(lower,lo,M*4,cudaMemcpyHostToDevice);cudaMemcpy(upper,up,M*4,cudaMemcpyHostToDevice);
 auto launch=[&](){lb_keogh<<<(n+255)/256,256,(M+256)*4>>>(subject,avgs,stds,out,lower,upper,M,N);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);double source_error=0;for(int i=0;i<n;i++){if(!std::isfinite(got[i])||got[i]<0)return 6;source_error=fmax(source_error,fabs((double)got[i]-source_ref[i]));}if(source_error>1e-2)return 7;printf("STUDY_SOURCE_CPU max_abs_error=%.12g\n",source_error);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.01));ok&=std::isfinite((double)got[j])&&e<=0.01;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
