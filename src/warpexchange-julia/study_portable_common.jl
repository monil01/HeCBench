using CUDA,Printf
function exchange_study(factory)
 host=reinterpret(Int32,read(joinpath(@__DIR__,"fixture_i32.bin")));expected=reinterpret(Int32,read(joinpath(@__DIR__,"expected_i32_guards.bin")));ns=(128,1152,4096,1048704);cases=[];prefix=Int32[];at=1
 for n in ns
  input=CuArray(host[at:at+n-1]);at+=n;output=CuArray(fill(Int32(0x13579bdf),n+16));f=factory(input,output,Int32(n));push!(cases,(input,output,f,n));append!(prefix,Array(input))
 end
 prefix==host||error("inputidentity");for c in cases;c[3]();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  result=Int32[];times=Float64[];sourceok=true
  for (j,c) in enumerate(cases)
   CUDA.device_synchronize(blocking=true);start=time_ns();for rep in 1:100;c[3]();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/100);v=Array(c[2]);append!(result,v)
   if j!=3
    for base in 0:128:c[4]-1,item in 0:3,lane in 0:30;sourceok&=v[base+4*(lane+1)+item+1]-v[base+4*lane+item+1]==1;end
   end
  end
  ok=result==expected&&sourceok;allok&=ok;err=maximum(abs.(Int64.(result).-Int64.(expected)));out=vcat(prefix,result);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"warpexchange_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"source_linearcase_difference_check_pass\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"internal_repeats\":100,\"jit_excluded\":true,\"timing_region\":\"sum_four_source_warppermutation_kernel_loopmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),string(sourceok),length(result),err,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
