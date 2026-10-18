
using CUDA,Printf
include(joinpath(@__DIR__,"study_math.jl"));include(joinpath(@__DIR__,"study_cases.jl"))
const G=16;const GUARD=12345.5f0
function run_study(launch)
 video=collect(read(joinpath(@__DIR__,"video_u8.bin")));offsets=collect(reinterpret(Int32,read(joinpath(@__DIR__,"offsets_i32.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=Any[];prefix=Float64[]
 for (n,offs)in CASES
  nb=cld(n,256);host=[fill(GUARD,(j==9 ? nb : n)+2G) for j in 1:9]
  for i in 0:n-1;host[7][i+17]=32.125f0+Float32(i%7)/16f0;host[8][i+17]=31.375f0-Float32(i%5)/32f0;host[4][i+17]=1f0/Float32(n);end
  hind=fill(Int32(1234567),69n+2G);hseed=vcat(fill(Int32(1234567),G),Int32.(1:n),fill(Int32(1234567),G));hobj=vcat(fill(Int32(1234567),G),offsets,fill(Int32(1234567),G));hi=vcat(fill(UInt8(251),G),video,fill(UInt8(251),G));dev=CuArray.(host);ind=CuArray(hind);seed=CuArray(hseed);obj=CuArray(hobj);im=CuArray(hi)
  for x in vcat(dev,Any[ind,seed,obj,im]);append!(prefix,Array(x));end
  push!(states,(n=Int32(n),offs=offs,host=host,dev=dev,hind=hind,hseed=hseed,hobj=hobj,hi=hi,ind=ind,seed=seed,obj=obj,im=im))
 end
 function reset(a)
  for j in 1:9;copyto!(a.dev[j],a.host[j]);end;copyto!(a.ind,a.hind);copyto!(a.seed,a.hseed);CUDA.device_synchronize()
 end
 for a in states;for k in Int32(1):Int32(3);launch(a,k);CUDA.device_synchronize();end;reset(a);end
 for trial in 1:5
  out=Float64[];times=Float64[];ok=true;maxerr=0.
  for a in states
   reset(a)
   for k in Int32(1):Int32(3)
    start=time_ns();launch(a,k);CUDA.device_synchronize();push!(times,(time_ns()-start)*1e-9);off=a.offs[k]
    for j in 1:9
     v=Array(a.dev[j]);n=length(v)-2G;ref=gold[off+1:off+n];e=abs.(Float64.(v[G+1:end-G]).-ref);ok&=all(isfinite,v)&&all(e .<=1e-4 .+3e-5.*abs.(ref));maxerr=max(maxerr,maximum(e));off+=n;ok&=all(==(GUARD),v[1:G])&&all(==(GUARD),v[end-G+1:end]);append!(out,v)
    end
    for x in (a.ind,a.seed)
     v=Array(x);n=length(v)-2G;ok&=v[G+1:end-G]==gold[off+1:off+n];off+=n;ok&=all(==(Int32(1234567)),v[1:G])&&all(==(Int32(1234567)),v[end-G+1:end]);append!(out,v)
    end
    ob=Array(a.obj);vi=Array(a.im);ok&=ob==a.hobj&&vi==a.hi;append!(out,ob);append!(out,vi)
   end
  end
  file=tempname()*".bin";write(file,vcat(prefix,out));@printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_nine_frame_five_kernel_pipelines_and_device_completion\",\"max_abs_error\":%.12g,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),sum(times),maxerr,file,length(prefix)+length(out),length(prefix)+length(out))
 end
end
