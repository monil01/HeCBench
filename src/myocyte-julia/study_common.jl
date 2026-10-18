
using CUDA,Printf
const BASE=dirname(@__FILE__);const ORACLE="/home/imo/HeCBench_julia/.porting-state/portable_julia/runtime/myocyte/libstudy_oracle.so"
const G=16;const R=20
function myocyte_study(factory)
 input=collect(reinterpret(Float32,read(joinpath(BASE,"study_probes_f32.bin"))))
 lens=(91,18,91,3);arrays=map(n->CuArray(fill(12345.5f0,n+G)),lens);launch=factory(arrays)
 for c=0:3;copyto!(arrays[1],1,input,c*111+1,91);copyto!(arrays[2],1,input,c*111+92,18);launch(Int32(trunc(input[c*111+110])));CUDA.device_synchronize(blocking=true);end
 for trial=1:5
  prefix=Float64[];values=Float64[];guards=Float64[];times=Float64[];ok=true;ma=0.;mr=0.;balance=0.;checked=0
  for c=0:3
   t=input[c*111+110];copyto!(arrays[1],1,input,c*111+1,91);copyto!(arrays[2],1,input,c*111+92,18);dt=0.
   for repeat=1:R;CUDA.device_synchronize(blocking=true);start=time_ns();launch(Int32(trunc(t)));CUDA.device_synchronize(blocking=true);dt+=(time_ns()-start)*1e-9;end
   push!(times,dt/R);actual=map(Array,arrays);append!(prefix,actual[1][1:91]);append!(prefix,actual[2][1:18]);append!(prefix,[Float64(t),Float64(trunc(t))])
   coupled=actual[3][1:91];ccall((:study_couple32,ORACLE),Cvoid,(Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float32}),actual[1],actual[2],coupled,actual[4])
   metrics=zeros(Float64,6);ccall((:study_probe_check,ORACLE),Cvoid,(Float64,Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float64}),t,actual[1],actual[2],actual[3],actual[4],coupled,metrics)
   ok &= metrics[1]==0;checked+=Int(metrics[2]);ma=max(ma,metrics[3]);mr=max(mr,metrics[4]);balance=max(balance,metrics[6])
   append!(values,actual[3][1:91]);append!(values,actual[4][1:3]);append!(values,coupled)
   for i=1:4;append!(guards,actual[i][lens[i]+1:end]);end
  end
  ok &= all(==(12345.5),guards);checked+=length(guards);out=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,out))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"diffusion_balance_roundoff_budget_ratio\":%.17g,\"timing_region\":\"sum4_mean_source91RHS_device_evaluator_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"full_solver_trajectory_verified\":false,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",checked,ma,mr,balance,join(times,","),path,length(out));flush(stdout);ok||error("Fullfinite91RHS/coupling/precisionoracle/guardsfailed")
 end
end
