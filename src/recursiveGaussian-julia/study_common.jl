
using CUDA,Printf
include(joinpath(@__DIR__,"study_math.jl"));include(joinpath(@__DIR__,"study_cases.jl"))
const G=16;const GUARD=UInt32(0xdeadbeef)
function run_study(rec,transpose)
 inputs=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"inputs_u32.bin"))));gold=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin"))));coeff=collect(reinterpret(Float32,read(joinpath(@__DIR__,"coefficients_f32.bin"))));states=Any[];prefix=UInt32[]
 for(w,h,mode,io,co,oo,sigma)in CASES
  n=w*h;host=vcat(fill(GUARD,G),inputs[io+1:io+n],fill(GUARD,G));input=CuArray(host);tmp=CUDA.fill(GUARD,n+2G);out=CUDA.fill(GUARD,n+2G);gp=GaussParms(coeff[co+1:co+12]...)
  for p in(input,tmp,out);append!(prefix,Array(p));end;append!(prefix,reinterpret(UInt32,coeff[co+1:co+12]));push!(states,(w=Int32(w),h=Int32(h),mode=mode,n=n,host=host,input=input,tmp=tmp,out=out,gp=gp,oo=oo))
 end
 function sequence(a)
  rec(a.input,a.tmp,a.w,a.h,a.gp);transpose(a.tmp,a.out,a.w,a.h);rec(a.out,a.tmp,a.h,a.w,a.gp);transpose(a.tmp,a.out,a.h,a.w)
 end
 for a in states;sequence(a);CUDA.device_synchronize();end
 for trial in 1:5
  artifact=copy(prefix);ok=true;maximum_error=0;total=0.
  for a in states
   fill!(a.tmp,GUARD);fill!(a.out,GUARD);CUDA.device_synchronize();start=time_ns();for r in 1:3;sequence(a);end;CUDA.device_synchronize();total+=(time_ns()-start)*1e-9/3;timedtmp=Array(a.tmp);timedout=Array(a.out);stages=Vector{UInt32}[]
   rec(a.input,a.tmp,a.w,a.h,a.gp);CUDA.device_synchronize();push!(stages,Array(a.tmp));transpose(a.tmp,a.out,a.w,a.h);CUDA.device_synchronize();push!(stages,Array(a.out));rec(a.out,a.tmp,a.h,a.w,a.gp);CUDA.device_synchronize();push!(stages,Array(a.tmp));transpose(a.tmp,a.out,a.h,a.w);CUDA.device_synchronize();push!(stages,Array(a.out));ok&=timedtmp==stages[3]&&timedout==stages[4];push!(stages,timedtmp);push!(stages,timedout);actualinput=Array(a.input);ok&=actualinput==a.host;append!(artifact,actualinput)
   for(j,v)in enumerate(stages)
    ok&=all(==(GUARD),v[1:G])&&all(==(GUARD),v[end-G+1:end]);refslot=(0,1,2,3,2,3)[j];ref=gold[a.oo+refslot*a.n+1:a.oo+(refslot+1)*a.n]
    for i in eachindex(v),ch in 0:3
     byte=(v[i]>>(8ch))&UInt32(255);push!(artifact,byte)
     if G<i<=G+a.n
      expected=(ref[i-G]>>(8ch))&UInt32(255);err=abs(Int(byte)-Int(expected));ok&=err<=2;maximum_error=max(maximum_error,err);a.mode==2&&(ok&=byte==0)
     end
    end
   end
  end
  file=tempname()*".bin";write(file,artifact);@printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_five_fourkernel_batches_R3_plus_completion_mean_per_pipeline\",\"max_abs_error\":%d,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),total,maximum_error,file,length(artifact),length(artifact))
 end
end
