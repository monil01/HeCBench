using CUDA,Printf
include("study_device.jl")
function clink_study(factory)
 q=512*257;pc=q+146;input=reinterpret(Float32,read(joinpath(@__DIR__,"study_inputs_f32.bin")));expected=reinterpret(Float64,read(joinpath(@__DIR__,"study_expected_f64.bin")));xs=[];ws=[];ys=[];launches=[]
 for a in 0:2
  x=CuArray(input[a*pc+1:a*pc+q]);w=CuArray(input[a*pc+q+1:(a+1)*pc]);y=CUDA.zeros(Float32,q);push!(xs,x);push!(ws,w);push!(ys,y);push!(launches,factory(x,w,y))
 end
 for f in launches;f();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  times=Float64[]
  for f in launches
   CUDA.device_synchronize(blocking=true);t=time_ns();for r in 1:10;f();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9/10)
  end
  got=vcat([Array(y)for y in ys]...);err=maximum(abs.(Float64.(got).-expected));rel=maximum(abs.(Float64.(got).-expected)./max.(abs.(expected),1e-5));ok=all(isfinite,got)&&all(abs.(Float64.(got).-expected).<=1e-5.+1e-5.*abs.(expected));allok&=ok;out=vcat(input,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"clink_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"internal_repeats\":10,\"jit_excluded\":true,\"timing_region\":\"three_LSTM_case_kernel_loop_means_sum\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),length(got),err,rel,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
