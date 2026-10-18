using CUDA,Printf
include("study_device.jl")
function tsa_study(factory)
 input=reinterpret(Float64,read(joinpath(@__DIR__,"fixture_f64.bin")));expected=reinterpret(Float64,read(joinpath(@__DIR__,"expected_f64.bin")));cases=[];prefix=Float64[];at=1
 for T in (Float32,Float64)
  for (w,h,steps) in ((64,96,3),(70,98,4),(34,50,1))
   n=w*h;a=T(input[at]);b=T(input[at+1]);at+=2;hr=T.(input[at:at+n-1]);at+=n;hi=T.(input[at:at+n-1]);at+=n
   rs=[CuArray(hr),CUDA.zeros(T,n)];ims=[CuArray(hi),CUDA.zeros(T,n)];launch=factory(a,b,Int32(w),Int32(h),rs,ims);push!(cases,(rs,ims,hr,hi,steps,launch));append!(prefix,[a,b]);append!(prefix,Float64.(Array(rs[1])));append!(prefix,Float64.(Array(ims[1])))
  end
 end
 prefix==input||error("inputidentity")
 function reset!(c)
  rs,ims,hr,hi,steps,f=c;copyto!(rs[1],hr);copyto!(ims[1],hi);fill!(rs[2],0);fill!(ims[2],0)
 end
 for c in cases
  reset!(c);for s in 0:c[5]-1;c[6](s%2+1);end
 end
 CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  result=Float64[];times=Float64[]
  for c in cases
   reset!(c);CUDA.device_synchronize(blocking=true);start=time_ns();for s in 0:c[5]-1;c[6](s%2+1);end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/c[5]);active=c[5]%2+1
   for b in (c[1][active],c[2][active],c[1][3-active],c[2][3-active]);append!(result,Float64.(Array(b)));end
  end
  errors=abs.(result.-expected);tolerances=vcat(fill(1e-5,58816),fill(1e-11,58816));ok=all(isfinite,result)&&all(errors.<=tolerances);allok&=ok;out=vcat(prefix,result);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"tsa_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"max_scaled_absolute_oracle_error\":%.12g,\"internal_repeats\":1,\"jit_excluded\":true,\"timing_region\":\"sum_six_evolution_perstep_means_pairedrotations_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),length(result),maximum(errors),maximum(errors./tolerances),file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
