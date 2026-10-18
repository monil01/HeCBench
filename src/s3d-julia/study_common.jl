using CUDA,Printf
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
