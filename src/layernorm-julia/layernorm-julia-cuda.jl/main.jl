using CUDA
include(joinpath(@__DIR__,"..","layernorm_math.jl"))
function ln0_kernel!(args,u)
 sh=CuStaticSharedArray(Float32,32)
 ln0!(Int32(threadIdx().x-1),Int32(blockIdx().x-1),Int32(blockDim().x),sh,args...,u)
 return
end
function lnwarp_kernel!(k,args)
 lnwarp_site!(k,Int32(threadIdx().x-1),Int32(blockIdx().x-1),Int32(blockDim().x),args...)
 return
end
function factory(c,algorithm,BT)
 if algorithm==0
  u=Val(Int(c.C)%4==0 ? 4 : 1);kernel=@cuda launch=false ln0_kernel!(c.args0,u)
  ()->kernel(c.args0,u;threads=BT,blocks=Int(c.N))
 else
  k=Val(algorithm);kernel=@cuda launch=false lnwarp_kernel!(k,c.argswarp)
  ()->kernel(k,c.argswarp;threads=BT,blocks=cld(Int(c.N)*32,BT))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
layernorm_study(factory)
