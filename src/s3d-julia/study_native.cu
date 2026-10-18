#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "../s3d-cuda/gr_base.h"
#include "../s3d-cuda/ratt.h"
#include "../s3d-cuda/ratt2.h"
#include "../s3d-cuda/ratx.h"
#include "../s3d-cuda/qssa.h"
#include "../s3d-cuda/qssa2.h"
#include "../s3d-cuda/rdwdot.h"
#define CK(x) do{auto s=(x);if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(2);}}while(0)
template<class T>T*alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void run(){const int N=std::is_same<T,float>::value?24*24*24:16*16*16,R=100;double atol=std::is_same<T,float>::value?8.*1.401298464324817e-45:1e-55,rtol=std::is_same<T,float>::value?2e-4:1e-10;std::vector<T> ht(N,T(1000)),hp(N,T(1.0132e6)),hy(22*N,T(0)),hm(22,T(1));for(int i=0;i<N;i++){hy[14*N+i]=T(.064);hy[3*N+i]=T(.218);hy[21*N+i]=T(.718);}T*gpu_t=alloc<T>(N),*gpu_p=alloc<T>(N),*gpu_y=alloc<T>(22*N),*gpu_wdot=alloc<T>(22*N),*gpu_molwt=alloc<T>(22),*gpu_rf=alloc<T>(206*N),*gpu_rb=alloc<T>(206*N),*gpu_rklow=alloc<T>(21*N),*gpu_c=alloc<T>(22*N),*gpu_a=alloc<T>(121*N),*gpu_eg=alloc<T>(32*N);CK(cudaMemcpy(gpu_t,ht.data(),N*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(gpu_p,hp.data(),N*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(gpu_y,hy.data(),22*N*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(gpu_molwt,hm.data(),22*sizeof(T),cudaMemcpyHostToDevice));T tconv=1,pconv=1,rateconv=1;auto launch=[&](){
 ratt_kernel<<<N/128,128>>>(gpu_t,gpu_rf,tconv);
 rdsmh_kernel<<<N/128,128>>>(gpu_t,gpu_eg,tconv);
 gr_base<<<N/128,128>>>(gpu_p,gpu_t,gpu_y,gpu_c,tconv,pconv);
 ratt2_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt3_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt4_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt5_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt6_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt7_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt8_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt9_kernel<<<N/128,128>>>(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
 ratt10_kernel<<<N/128,128>>>(gpu_t,gpu_rklow,tconv);
 ratx_kernel<<<N/64,64>>>(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv);
 ratxb_kernel<<<N/64,64>>>(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv);
 ratx2_kernel<<<N/128,128>>>(gpu_c,gpu_rf,gpu_rb);
 ratx4_kernel<<<N/128,128>>>(gpu_c,gpu_rf,gpu_rb);
 qssa_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_a);
 qssab_kernel<<<N/64,64>>>(gpu_rf,gpu_rb,gpu_a);
 qssa2_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_a);
 rdwdot_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot2_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot3_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot6_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot7_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot8_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot9_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
 rdwdot10_kernel<<<N/128,128>>>(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
};launch();CK(cudaDeviceSynchronize());std::ifstream f(std::is_same<T,float>::value?"study_cpu_f32_f64.bin":"study_cpu_f64_f64.bin",std::ios::binary);std::vector<double> expected(508);f.read((char*)expected.data(),508*8);if(!f)exit(3);std::vector<double> prefix;for(auto*a:std::initializer_list<std::vector<T>*>{&ht,&hp,&hy,&hm})for(T x:*a)prefix.push_back(x);for(double a:{double(N),double(R),64.,128.,double(tconv),double(pconv),double(rateconv)})prefix.push_back(a);bool all=true;
for(int trial=1;trial<=5;trial++){CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<R;r++)launch();CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/R;std::vector<double> out=prefix;bool ok=true;double ma=0,mr=0;size_t offset=0;for(auto a:std::initializer_list<std::pair<T*,int>>{{gpu_wdot,22},{gpu_c,22},{gpu_rf,206},{gpu_rb,206},{gpu_rklow,21},{gpu_eg,31}}){std::vector<T> got(a.second*N);CK(cudaMemcpy(got.data(),a.first,got.size()*sizeof(T),cudaMemcpyDeviceToHost));for(int j=0;j<a.second;j++)for(int i=0;i<N;i++){double z=got[j*N+i],e=expected[offset+j],err=abs(z-e);ok&=std::isfinite(z)&&err<=atol+rtol*abs(e);ma=std::max(ma,err);mr=std::max(mr,err/std::max(abs(e),1e-300));out.push_back(z);}offset+=a.second;}all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/output_trial"+std::to_string(trial)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)out.data(),out.size()*8);o.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"error_metric_domain\":\"heterogeneous_full_WDOT_C_RF_RB_RKLOW_EG_arrays_sourceformula_CPU_not_independent_chemistry\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"device_kernel_stages\":27,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",508*N,ma,mr,path.c_str(),out.size());fflush(stdout);}if(!all)exit(1);}
int main(int argc,char**argv){if(argc>1&&std::string(argv[1])=="f64")run<double>();else run<float>();}
