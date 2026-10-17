const HH=8732
const WW=81
@inline function study_value(t,p,g,gn,w,m,q,x)
 @inbounds begin
  idx=q[x];tmp=m[x]!=0 ? -(g[x]+gn[x]) : -g[x];tmp=t(tmp*w[x]);sv=Float32(t(tmp*p[Int(idx)*HH+x]))
 end
 return tmp,sv,idx
end
@inline study_exp(x::Float16)=Float32(Float16(exp(Float32(x))))
@inline study_exp(x::Float32)=exp(x)
@inline study_exp(x::Float64)=exp(x)
function fixture(::Type{T}) where T
 p=T[-Float64(mod(13i+7,129)+1)/32 for i in 0:WW*HH-1]
 g=T[Float64(mod(i,13)+1)/32 for i in 0:HH-1];gn=T[Float64(mod(i,7)+1)/64 for i in 0:HH-1];w=T[Float64(mod(i,5)+1)/8 for i in 0:HH-1]
 m=Int64[mod(i,2) for i in 0:HH-1];q=Int64[mod(17i+3,WW) for i in 0:HH-1];want=Vector{T}(undef,WW*HH)
 @inbounds for x in 1:HH
  tmp,sv,idx=study_value(T,p,g,gn,w,m,q,x)
  for c in 0:WW-1
   off=c*HH+x;z=Float32(study_exp(p[off])*sv);want[off]=T(c==idx ? Float32(tmp)-z : -z)
  end
 end
 return p,g,gn,w,m,q,want
end
function study(run!)
 cases=[]
 for T in (Float16,Float32,Float64)
  p,g,gn,w,m,q,want=fixture(T);out=CUDA.zeros(T,WW*HH);args=(CuArray(p),CuArray(g),CuArray(gn),CuArray(w),CuArray(m),CuArray(q),out,T)
  run!(args...);CUDA.synchronize();push!(cases,(args,want))
 end
 for trial in 1:5
  total=0.;got=Float64[];wantall=Float64[]
  for (args,want) in cases
   for k in 1:10
    CUDA.synchronize();t=time_ns();run!(args...);CUDA.synchronize();total+=(time_ns()-t)*1e-9/10
   end
   append!(got,Float64.(Array(args[7])));append!(wantall,Float64.(want))
  end
  err=maximum(abs.(got.-wantall));ok=all(isfinite,got)&&err<=4e-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_three_precision_mean_kernels\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("CPU mixed-precision oracle failed")
 end
end
