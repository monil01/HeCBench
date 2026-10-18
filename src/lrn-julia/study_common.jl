
const BASE=dirname(@__FILE__);const G=32;const R=10
const DIMS=[(2,7,3,4,5),(1,4,1,3,7)]
function lrn_study(factory)
 states=map(enumerate(DIMS))do (c,dims)
  n=prod(dims);input=collect(reinterpret(Float32,read(joinpath(BASE,"study_case$(c)_f32.bin"))));ref=collect(reinterpret(Float64,read(joinpath(BASE,"oracle_case$(c)_f64.bin"))))
  src=CuArray(vcat(input[1:n],fill(12345.5f0,G)));up=CuArray(vcat(input[n+1:end],fill(12345.5f0,G)));fwd=CuArray(vcat(zeros(Float32,n),fill(12345.5f0,G)));bwd=CuArray(vcat(zeros(Float32,n),fill(12345.5f0,G)))
  launch=factory(src,up,fwd,bwd,dims)
  for wg in (64,128,256,512),mode=1:2;launch(wg,mode);end
  (n,src,up,fwd,bwd,ref,launch)
 end
 CUDA.device_synchronize(blocking=true)
 for trial=1:5
  prefix=Float64[];values=Float64[];guards=Float64[];times=Float64[];ma=0.;mr=0.;ok=true;mismatch=0
  for (c,(n,src,up,fwd,bwd,ref,launch))in enumerate(states)
   for d in (src,up)
    actual=Array(d);append!(prefix,Float64.(reinterpret(UInt32,actual[1:n])));append!(guards,Float64.(actual[n+1:end]))
   end
   append!(prefix,[DIMS[c]...,5,R,c,5]);append!(prefix,Float64.(reinterpret(UInt32,Float32[.000122,.75,1.])))
   for wg in (64,128,256,512),mode=1:2
    CUDA.device_synchronize(blocking=true);t=time_ns();for _=1:R;launch(wg,mode);end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9/R)
    out=Array(mode==1 ? fwd : bwd);append!(values,Float64.(out[1:n]))
    for i=1:n
     q=ref[(mode-1)*n+i];d=abs(Float64(out[i])-q);ma=max(ma,d);mr=max(mr,d/max(abs(q),1e-30));pass=isfinite(out[i])&&d<=8e-7+4e-6*abs(q);mismatch+=!pass;ok &= pass
    end
   end
   for d in (fwd,bwd);append!(guards,Float64.(Array(d)[n+1:end]));end
  end
  ok &= all(==(12345.5),guards);out=vcat(prefix,values,guards);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR","/tmp"),"trial$(trial).bin");write(path,reinterpret(UInt8,out))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":7648,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"mismatch_values\":%d,\"timing_region\":\"sum16_lrn_fwd_bwd_four_workgroups_two_cases_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ma,mr,mismatch,join(times,","),path,length(out));flush(stdout);ok || error("fullLRN/gradient/guard check failed")
 end
end
