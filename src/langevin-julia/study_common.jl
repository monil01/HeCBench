
const BASE=dirname(@__FILE__);const N=4096;const G=32;const R=20
function langevin_study(factory)
 inputs=[collect(reinterpret(Float32,read(joinpath(BASE,"study_case$(c)_f32.bin")))) for c=1:2]
 oracles=[collect(reinterpret(Float64,read(joinpath(BASE,"oracle_case$(c)_f64.bin")))) for c=1:2]
 states=map(inputs)do input
  a=CuArray(vcat(input,fill(12345.5f0,G)));outs=[CUDA.fill(12345.5f0,N+G) for _=1:3];f=factory(a,outs)
  for mode=1:3;f(mode);end
  (a,outs,f)
 end
 CUDA.device_synchronize(blocking=true)
 for trial=1:5
  prefix=Float64[];values=Float64[];guards=Float64[];times=Float64[];ma=0.;mr=0.;ratio=0.;ok=true
  for c=1:2
   a,outs,launch=states[c];actual=Array(a);append!(prefix,Float64.(reinterpret(UInt32,actual[1:N])));append!(prefix,[N,R,256,16,c,3,0,0]);append!(guards,Float64.(actual[N+1:end]))
   for mode=1:3
    CUDA.device_synchronize(blocking=true);t=time_ns();for _=1:R;launch(mode);end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9/R)
    out=Array(outs[mode]);append!(values,Float64.(out[1:N]));append!(guards,Float64.(out[N+1:end]))
    for i=1:N
     expected=oracles[c][(mode-1)*N+i];diff=abs(Float64(out[i])-expected);budget=mode==3 ? 2e-7 : max(2e-6,8*2.0^-23/abs(Float64(inputs[c][i])))
     ma=max(ma,diff);mr=max(mr,diff/max(abs(expected),1e-30));ratio=max(ratio,diff/budget);ok &= isfinite(out[i]) && diff<=budget
    end
   end
  end
  ok &= all(==(12345.5),guards);allout=vcat(prefix,values,guards);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR","/tmp"),"trial$(trial).bin");write(path,reinterpret(UInt8,allout))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":24832,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"max_policy_budget_ratio\":%.17g,\"timing_region\":\"sum6_langevin_k0_k1_k2_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",ma,mr,ratio,join(times,","),path,length(allout));flush(stdout);ok || error("fullpolicy/finite/guard checker failed")
 end
end
