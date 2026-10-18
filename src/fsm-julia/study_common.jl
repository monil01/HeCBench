using CUDA,Printf
include("study_math.jl")
function fsm_study(factory)
 ns=(257,4099,1021);host=reinterpret(UInt16,read(joinpath(@__DIR__,"fixture_u16.bin")));expected=reinterpret(UInt64,read(joinpath(@__DIR__,"expected_u64.bin")));inputs=[];prefix=UInt64[];at=1
 for n in ns
  d=CuArray(host[at:at+n-1]);at+=n;push!(inputs,d);append!(prefix,UInt64.(Array(d)))
 end
 best=CUDA.zeros(Int32,19);winner=reinterpret(UInt64,view(best,1:2));rng=CuArray{UInt32}(undef,262144);bfsm=CuArray{UInt8}(undef,16384);same=CuArray{UInt8}(undef,1024);smax=CuArray{Int32}(undef,1024);sbest=similar(smax);old=similar(smax)
 states=CuArray{UInt32}(undef,8192*262144) #8GiB explicit lane-private table backing, allocated outside ROI.
 fs=[factory(Int32(ns[j]),inputs[j],best,winner,rng,bfsm,same,smax,sbest,old,states) for j in 1:3]
 reset!()=CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(best)),UInt8(0),sizeof(best))
 for f in fs;reset!();f();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  result=UInt64[];times=Float64[]
  for f in fs
   CUDA.device_synchronize(blocking=true);start=time_ns()
   for rep in 1:10;reset!();f();end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/10)
   for b in (best,rng,bfsm,same,smax,sbest,old);append!(result,UInt64.(Array(b)));end
  end
  ok=result==expected;allok&=ok;err=maximum(abs.(Int64.(result).-Int64.(expected)));out=vcat(prefix,result);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"fsm_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%s,\"internal_repeats\":10,\"jit_excluded\":true,\"timing_region\":\"sum_three_trace_source_repeatmeans_reset_GA_Max_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),length(result),err,ok ? "0" : "null",file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
