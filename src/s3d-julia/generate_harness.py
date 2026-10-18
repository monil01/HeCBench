from pathlib import Path
import json
R=Path('/home/imo/HeCBench_julia');P=R/'src/s3d-julia';meta=json.load(open(P/'translation_map.json'));stages=meta['pipeline'];fields='N,gpu_t,gpu_p,gpu_y,gpu_wdot,gpu_molwt,gpu_rf,gpu_rb,gpu_rklow,gpu_c,gpu_a,gpu_eg,tconv,pconv,rateconv'
cpp=r'''#include <cuda_runtime.h>
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
'''
for st in stages:cpp+=f" {st['name']}<<<N/{st['threads']},{st['threads']}>>>({','.join(st['args'])});\n"
cpp+=r'''};launch();CK(cudaDeviceSynchronize());std::ifstream f(std::is_same<T,float>::value?"study_cpu_f32_f64.bin":"study_cpu_f64_f64.bin",std::ios::binary);std::vector<double> expected(508);f.read((char*)expected.data(),508*8);if(!f)exit(3);std::vector<double> prefix;for(auto*a:std::initializer_list<std::vector<T>*>{&ht,&hp,&hy,&hm})for(T x:*a)prefix.push_back(x);for(double a:{double(N),double(R),64.,128.,double(tconv),double(pconv),double(rateconv)})prefix.push_back(a);bool all=true;
for(int trial=1;trial<=5;trial++){CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<R;r++)launch();CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/R;std::vector<double> out=prefix;bool ok=true;double ma=0,mr=0;size_t offset=0;for(auto a:std::initializer_list<std::pair<T*,int>>{{gpu_wdot,22},{gpu_c,22},{gpu_rf,206},{gpu_rb,206},{gpu_rklow,21},{gpu_eg,31}}){std::vector<T> got(a.second*N);CK(cudaMemcpy(got.data(),a.first,got.size()*sizeof(T),cudaMemcpyDeviceToHost));for(int j=0;j<a.second;j++)for(int i=0;i<N;i++){double z=got[j*N+i],e=expected[offset+j],err=abs(z-e);ok&=std::isfinite(z)&&err<=atol+rtol*abs(e);ma=std::max(ma,err);mr=std::max(mr,err/std::max(abs(e),1e-300));out.push_back(z);}offset+=a.second;}all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/output_trial"+std::to_string(trial)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)out.data(),out.size()*8);o.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"error_metric_domain\":\"heterogeneous_full_WDOT_C_RF_RB_RKLOW_EG_arrays_sourceformula_CPU_not_independent_chemistry\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"device_kernel_stages\":27,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",508*N,ma,mr,path.c_str(),out.size());fflush(stdout);}if(!all)exit(1);}
int main(int argc,char**argv){if(argc>1&&std::string(argv[1])=="f64")run<double>();else run<float>();}
''';(P/'study_native.cu').write_text(cpp)
(P/'study_common.jl').write_text(r'''using CUDA,Printf
include("study_device.jl")
function s3d_study(factory)
 fp64=ARGS[1]=="f64";FT=fp64 ? Float64 : Float32;N=Int32(fp64 ? 16^3 : 24^3);R=100;atol=fp64 ? 1e-55 : 8.0*Float64(nextfloat(0.0f0));rtol=fp64 ? 1e-10 : 2e-4
 ht=fill(FT(1000),N);hp=fill(FT(1.0132e6),N);hy=zeros(FT,22*N);for i in 1:N;hy[14*N+i]=FT(.064);hy[3*N+i]=FT(.218);hy[21*N+i]=FT(.718);end;hm=ones(FT,22)
 gpu_t=CuArray(ht);gpu_p=CuArray(hp);gpu_y=CuArray(hy);gpu_molwt=CuArray(hm);gpu_wdot=CuArray{FT}(undef,22*N);gpu_rf=CuArray{FT}(undef,206*N);gpu_rb=similar(gpu_rf);gpu_rklow=CuArray{FT}(undef,21*N);gpu_c=CuArray{FT}(undef,22*N);gpu_a=CuArray{FT}(undef,121*N);gpu_eg=CuArray{FT}(undef,32*N);tconv=one(FT);pconv=one(FT);rateconv=one(FT)
 launch=factory(N,gpu_t,gpu_p,gpu_y,gpu_wdot,gpu_molwt,gpu_rf,gpu_rb,gpu_rklow,gpu_c,gpu_a,gpu_eg,tconv,pconv,rateconv);launch();CUDA.device_synchronize(blocking=true);expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"study_cpu_$(ARGS[1])_f64.bin"))));@assert length(expected)==508
 prefix=vcat(Float64.(ht),Float64.(hp),Float64.(hy),Float64.(hm),Float64[N,R,64,128,tconv,pconv,rateconv]);allok=true
 for trial in 1:5
  CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:R;launch();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/R
  payload=vcat(Float64.(Array(gpu_wdot)),Float64.(Array(gpu_c)),Float64.(Array(gpu_rf)),Float64.(Array(gpu_rb)),Float64.(Array(gpu_rklow)),Float64.(Array(gpu_eg)[1:31*N]));err=abs.(reshape(payload,Int(N),508).-reshape(expected,1,508));ok=all(isfinite,payload)&&all(err.<=atol.+rtol.*reshape(abs.(expected),1,508));allok&=ok;ma=maximum(err);mr=maximum(err./reshape(max.(abs.(expected),1e-300),1,508));out=vcat(prefix,payload);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"error_metric_domain\":\"heterogeneous_full_WDOT_C_RF_RB_RKLOW_EG_arrays_sourceformula_CPU_not_independent_chemistry\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"device_kernel_stages\":27,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(payload),ma,mr,path,length(out))
 end
 allok||error("sourceformulaCPU/fullstate mismatch");println("PASS")
end
''')
for tag,N,atol,rtol in [('f32',24**3,8*1.401298464324817e-45,2e-4),('f64',16**3,1e-55,1e-10)]:
 assets=[f'study_cpu_{tag}_f64.bin'];cfg=dict(benchmark='s3d',gpu=6,timeout=1500,inputs_shared_verified=True,timed_regions_equivalent_verified=True,atol=atol,rtol=rtol,native_reference_repaired=False,configuration_id=f's3d_originalclass1_{tag}_N{N}_R100_all27stages_v1',input_prefix_count=24*N+29,input_contract=f'Originalclass1 {tag} N{N} uniformT1000/P1.0132e6,Yindices3=.218/14=.064/21=.718,molwtallones; conversionfactors1. Complete27stagepipeline,source100passes pertrial;1fullpipelinewarmup then5trialsmeanlast4 averageperpipeline. Globalcompletion; input/allocation/outputcheckingoutsideROI; noresetrequiredsourceoverwrites. FullWDOT22/C22/RF206/RB206/RKLOW21/EG31arrays,508Nvalues; unusedEG32/Aworkspaceexcluded. SourceformulaCPU508components are computationalcrosscheck not independentchemicalmechanism. Float32atol8smallestsubnormal/rtol2e-4; Float64atol1e-55/rtol1e-10predeclared. SourceFLT_MIN preservedasFloat32floor inbothprecisions; nofastmath/FTZrequested. NoODE/flowintegration/modelphysicalvalidityclaim;T>1000branchunexercised.',models={})
 cfg['models']['cuda']=dict(cwd=str(P),build=['nvcc','-O3','-std=c++17','-arch={arch}','study_native.cu','-o','study_native'],command=['./study_native',tag],structured=True,runtime_assets=assets,source_dependencies=['../s3d-cuda/'+q for q in ['S3D.h','cudacommon.h']+list(meta['source_headers'])]+assets,verification_method='full_defined_rate_and_WDOT_arrays_sourceformula_CPU_plus_dualreference')
 d=P/'s3d-julia-cuda.jl';cfg['models']['cuda_jl']=dict(cwd=str(d),command=['julia',f'--project={R}/src/_julia_ka_env',str(d/'main.jl'),tag],structured=True,source_dependencies=['../study_device.jl','../study_common.jl','../'+assets[0]],verification_method='full_defined_rate_and_WDOT_arrays_sourceformula_CPU_plus_dualreference');(P/f'study_config_{tag}_refs.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('native/common/two configs ready')
