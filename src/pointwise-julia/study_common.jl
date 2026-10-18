using Printf
include("study_cases.jl")
const G=16;const GUARD=12345.5f0
function run_study(launch)
 inputs=collect(reinterpret(Float32,read(joinpath(@__DIR__,"inputs_f32.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=Any[];prefix=Float64[]
 for (s,l,h,b,mode,sizes,offs,go,groups) in CASES
  payloads=[fill(GUARD,n) for n in sizes]
  for (k,o) in zip((3,4,5,6),offs);payloads[k]=inputs[o+1:o+sizes[k]];end
  host=[vcat(fill(GUARD,G),v,fill(GUARD,G)) for v in payloads];dev=[CuArray(v) for v in host]
  for v in dev;append!(prefix,Array(v));end
  push!(states,(s=s,l=l,h=h,b=b,mode=mode,sizes=sizes,host=host,dev=dev,go=go,groups=groups))
 end
 function sequence(a)
  ne=a.h*a.b;v=[view(x,G+1:length(x)-G) for x in a.dev];hh,ii,cc,bs,th,ti,lg=v;total=0.
  for (ls,le,rs,re) in a.groups
   start=time_ns()
   for layer in ls:le-1,t in rs:re-1
    launch(a.h,a.b,view(th,4*layer*ne+1:4*(layer+1)*ne),view(ti,4*t*ne+1:4*(t+1)*ne),view(bs,8*layer*a.h+1:8*(layer+1)*a.h),view(lg,4*(t*ne+layer*a.s*ne)+1:4*(t*ne+layer*a.s*ne+ne)),view(hh,(t+1)*ne+layer*(a.s+1)*ne+1:(t+2)*ne+layer*(a.s+1)*ne),view(ii,t*ne+(layer+1)*a.s*ne+1:(t+1)*ne+(layer+1)*a.s*ne),view(cc,t*ne+layer*(a.s+1)*ne+1:(t+1)*ne+layer*(a.s+1)*ne),view(cc,(t+1)*ne+layer*(a.s+1)*ne+1:(t+2)*ne+layer*(a.s+1)*ne))
   end
   CUDA.device_synchronize();total+=(time_ns()-start)*1e-9
  end
  return total
 end
 for a in states;sequence(a);end
 for trial in 1:5
  out=Float64[];times=Float64[];correct=true;maxerr=0.
  for a in states
   for k in eachindex(a.dev);copyto!(a.dev[k],a.host[k]);end;CUDA.device_synchronize();push!(times,sequence(a));off=a.go
   for k in eachindex(a.dev)
    v=Array(a.dev[k]);ref=gold[off+1:off+length(v)];off+=length(v);e=abs.(Float64.(v).-ref);correct &= all(isfinite,v)&&all(e .<=2e-5 .+2e-6.*abs.(ref));maxerr=max(maxerr,maximum(e));correct &= all(==(GUARD),v[1:G])&&all(==(GUARD),v[end-G+1:end]);append!(out,v)
    if k in (4,5,6);correct &= v==a.host[k];end
   end
  end
  file=tempname()*".bin";open(file,"w")do io;write(io,vcat(prefix,out));end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum_five_source_wavefront_group_launch_windows_plus_device_completion\",\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,sum(times),string(correct),maxerr,join(times,","),file,length(prefix)+length(out),length(prefix)+length(out))
 end
end
