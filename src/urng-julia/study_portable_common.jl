using CUDA,Printf
include("study_device.jl")
function urng_study(factory)
 raw=read(joinpath(@__DIR__,"fixture_rgba.bin"));expected=read(joinpath(@__DIR__,"expected_rgba_guards.bin"));host=reinterpret(NTuple{4,UInt8},raw);ns=(16384,1031);cases=[];prefix=UInt8[];at=1
 for n in ns
  input=CuArray(host[at:at+n-1]);at+=n;output=CuArray(fill(ntuple(_->UInt8(165),4),n+17));f=factory(input,output,Int32(n));push!(cases,(input,output,f,n));append!(prefix,reinterpret(UInt8,Array(input)))
 end
 prefix==raw||error("inputidentity");for c in cases;c[3]();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  result=UInt8[];times=Float64[];means=Float64[];offset=1
  for c in cases
   CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:1000;c[3]();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/1000)
   values=collect(reinterpret(UInt8,Array(c[2])));nbytes=4c[4];push!(means,sum(Int(values[k])-Int(raw[offset+k-1]) for k in 1:nbytes)/(nbytes*25));offset+=nbytes;append!(result,values)
  end
  ok=result==expected;allok&=ok;err=maximum(abs.(Int.(result).-Int.(expected)));countbad=count(k->result[k]!=expected[k],eachindex(result))
  out=vcat(prefix,result);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"urng_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g],\"correct\":%s,\"source_mean_check_pass\":%s,\"source_means\":[%.12g,%.12g],\"mismatched_values\":%d,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"internal_repeats\":1000,\"jit_excluded\":true,\"timing_region\":\"sum_two_source_kernel_loopmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),string(all(abs.(means).<1)),means...,countbad,length(result),err,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
