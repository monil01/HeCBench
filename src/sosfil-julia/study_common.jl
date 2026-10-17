
using CUDA,Printf
function sos_study(launch!)
 prefix=reinterpret(Float64,read("study_input_f64.bin"));gold=reinterpret(Float64,read("study_expected_f64.bin"));states=[];pos=1;gpos=1
 for bits in (32,64),kind=0:1
  b,signals,samples,sections,ziwidth,shared=Int.(prefix[pos:pos+5]);pos+=6;@assert b==bits&&sections==64&&ziwidth==2;T=bits==32 ? Float32 : Float64;sos=T.(prefix[pos:pos+383]);pos+=384;zi=T.(prefix[pos:pos+signals*128-1]);pos+=signals*128;x=T.(prefix[pos:pos+signals*samples-1]);pos+=signals*samples;g=gold[gpos:gpos+3*length(x)-1];gpos+=3*length(x);ds=CuArray(sos);dz=CuArray(zi);dx=CuArray(x);template=CUDA.zeros(T,shared);push!(states,(signals,samples,shared,x,ds,dz,dx,template,g));launch!(signals,samples,Val(shared),ds,dz,dx,template)
 end
 CUDA.synchronize();allok=true
 for trial=1:5
  captured=[]
  for (signals,samples,shared,x,s,z,dx,template,g) in states
   copyto!(dx,x);vals=Float64[]
   for _=1:2;launch!(signals,samples,Val(shared),s,z,dx,template);CUDA.synchronize();append!(vals,Float64.(Array(dx)));end
   push!(captured,vals)
  end
  times=Float64[]
  for (signals,samples,shared,x,s,z,dx,template,g) in states
   CUDA.synchronize();start=time_ns()
   for _=1:3;launch!(signals,samples,Val(shared),s,z,dx,template);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/3)
  end
  ok=true;maxerr=0.;out=collect(prefix)
  for (i,(signals,samples,shared,x,s,z,dx,template,g)) in enumerate(states)
   append!(captured[i],Float64.(Array(dx)));v=captured[i];e=abs.(v.-g);maxerr=max(maxerr,maximum(e));ok&=all(isfinite,v)&&all(e.<=1e-4.+1e-4.*abs.(g));append!(out,v)
  end
  allok&=ok;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":5406,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",maxerr,path,length(out))
 end
 println(allok ? "PASS" : "FAIL")
end
