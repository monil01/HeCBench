# Derivedfromstudy_common.jl; differentframeworkscalar-returnidiom suppliedexplicitly.
using CUDA,Printf
include("study_device.jl")
function dp_study(factory,idiom_factory,mode=parse(Int,ARGS[1]))
 data=reinterpret(Int32,read(joinpath(@__DIR__,"study_inputs_i32.bin")));cases=[];prefix=Float64[]
 for T in (Float32,Float64)
  at=1
  for z in 1:2
   n=Int(data[at]);pad=Int(data[at+1]);at+=2;a=T.(data[at:at+pad-1]);at+=pad;b=T.(data[at:at+pad-1]);at+=pad;expected=sum(Int64.(a).*Int64.(b));sum(abs.(Int64.(a).*Int64.(b)))<2^24||error("FP32bound");da=CuArray(a);db=CuArray(b);dr=CUDA.zeros(T,1);blocks=pad÷4096
   custom=factory(da,db,dr,pad,blocks);handle=CUDA.CUBLAS.handle();vendor=()->CUDA.CUBLAS.cublasDotEx(handle,n,da,T,1,db,T,1,dr,T,T);va=view(da,1:n);vb=view(db,1:n);idiom=idiom_factory(da,db,dr,n,pad,blocks,custom)
   launch=mode==1 ? custom : mode==2 ? vendor : idiom;push!(cases,(da,db,dr,launch,expected));append!(prefix,[n,pad]);append!(prefix,Float64.(Array(da)));append!(prefix,Float64.(Array(db)))
  end
 end
 # Native custombranch zeroes scalar ineachrepeat; useactualdriver memset onkernel taskstream.
 memset!(r)=CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(r)),UInt8(0),sizeof(eltype(r)))
 for (a,b,r,f,e) in cases
  for rep in 1:100;if mode==1;memset!(r);end;f();end
 end
 CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  times=Float64[];values=Float64[]
  for (a,b,r,f,e) in cases
   CUDA.device_synchronize(blocking=true);t=time_ns();lastvalue=zero(eltype(r))
   for rep in 1:20
    if mode==1;memset!(r);end
    v=f();if mode==3;lastvalue=v;end
   end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-t)*1e-9/20);push!(values,mode==3 ? Float64(lastvalue) : Float64(only(Array(r))))
  end
  expected=[x[5] for x in cases];err=maximum(abs.(values.-expected));ok=all(isfinite,values)&&values==expected;allok&=ok;out=vcat(prefix,values);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"dp_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":4,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"internal_repeats\":20,\"jit_excluded\":true,\"timing_region\":\"four_FP32_FP64_named_dot_cases_meanloop_sum\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),times...,string(ok),err,err,file,length(out))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
