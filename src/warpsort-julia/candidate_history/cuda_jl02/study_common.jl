using Printf
CUDA.allowscalar(false)
include("cases.jl")
function guarded(n,::Type{T}) where T
 raw=CUDA.fill(UInt32(0xa5a5a5a5),n+64)
 payload=unsafe_wrap(CuArray,CuPtr{T}(pointer(raw)+128),(n,);own=false)
 return raw,payload
end
function runblock(trial,emit)
 prefix=Float64[];outputs=Float64[];elapsed=0.0
 for (n,mode,bs) in CASES
  ar,a=guarded(n,Float32);br,b=guarded(n,Float32);ir,ix=guarded(n,Int32)
  copyto!(a,reinterpret(Float32,bs));fill!(b,0f0);fill!(ix,Int32(0));CUDA.device_synchronize()
  ai=Array(ar);@assert ai[33:n+32]==bs
  append!(prefix,(n,mode));append!(prefix,ai[33:n+32])
  CUDA.device_synchronize();t=time_ns();launchcase(a,b,ix,n,mode);CUDA.device_synchronize();elapsed+=(time_ns()-t)*1e-9
  ah=Array(ar);bh=Array(br);ih=Array(ir);@assert ah[33:n+32]==bs
  ob=bh[33:n+32];vals=reinterpret(Float32,ob)
  @assert all(vals[i]>=vals[i+1] for i in 1:n-1)
  normbits(x)=(x&0x7fffffff)==0 ? UInt32(0) : x
  @assert sort(normbits.(ob))==sort(normbits.(bs))
  append!(outputs,ob)
  if mode==1
   want=sortperm(1:n;lt=(i,j)->(reinterpret(Float32,bs[i])>reinterpret(Float32,bs[j]) || (reinterpret(Float32,bs[i])==reinterpret(Float32,bs[j])&&i>j)))
   got=reinterpret(Int32,ih[33:n+32]);@assert got==Int32.(want.-1);@assert ob==bs[want]
   append!(outputs,got)
  end
  for raw in (ah,bh,ih)
   guards=vcat(raw[1:32],raw[n+33:n+64]);append!(outputs,guards);@assert all(==(UInt32(0xa5a5a5a5)),guards)
  end
 end
 if emit
  append!(prefix,outputs);@assert length(prefix)==91402
  name="trial$(trial).bin";open(name,"w")do io;write(io,prefix);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":91402,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"sum258_source_singlewarp_kernel_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":91402,\"byte_order\":\"little\"}}\n",trial,elapsed,name)
 end
 return
end
runblock(0,false)
for t in 1:5;runblock(t,true);end
