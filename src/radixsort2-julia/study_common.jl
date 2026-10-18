using CUDA,Printf
include("study_device.jl")
function radixsort_study(factory)
 mode=isempty(ARGS) ? "uint_pairs" : ARGS[1];fp=startswith(mode,"float");only=endswith(mode,"keys");N=Int32(1048576);REPS=100;T=fp ? Float32 : UInt32;tag=fp ? "f32" : "u32";input=collect(reinterpret(T,read(joinpath(@__DIR__,"study_input_$(tag).bin"))));expected=collect(reinterpret(T,read(joinpath(@__DIR__,"study_expected_keys_$(tag).bin"))));values=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"study_input_values_u32.bin"))));keys=CuArray(input);vals=only ? CUDA.zeros(UInt32,0) : CuArray(values);launch=factory(keys,vals,N,only)
 reset()=(copyto!(keys,input);if !only;copyto!(vals,values);end;nothing)
 reset();CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);prefix=Float64.(input);if !only;append!(prefix,Float64.(values));end;append!(prefix,Float64[N,only,fp,REPS,32]);allok=true
 for trial in 1:5
  seconds=0.
  for rep in 1:REPS
   reset();CUDA.device_synchronize(blocking=true);start=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds+=(time_ns()-start)*1e-9/REPS
  end
  gotkeys=Array(keys);gotvals=Array(vals);ok=gotkeys==expected&&all(isfinite,gotkeys);ma=maximum(abs.(Float64.(gotkeys).-Float64.(expected)));mr=maximum(abs.(Float64.(gotkeys).-Float64.(expected))./max.(abs.(Float64.(expected)),1e-30))
  if !only
   ok&=sort(gotvals)==values
   for i in 1:Int(N);id=Int(gotvals[i])+1;ok&=1<=id<=N&&(gotkeys[i]==input[id]);end
  end
  allok&=ok;out=vcat(prefix,Float64.(gotkeys));if !only;append!(out,Float64.(gotvals));end;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"error_metric_domain\":\"sorted_keys_exact_plus_pair_permutation_association_properties\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(out),ma,mr,REPS,path,length(out))
 end
 allok||error("Sortkeys/pairassociationmismatch");println("PASS")
end
