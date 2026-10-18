using CUDA,JACC
include(joinpath(@__DIR__,"..","layernorm_math.jl"))
const LN_TEMPLATE=CUDA.zeros(Float32,32)
function ln0_kernel!(i,args,u,nt,template)
 sh=JACC.shared(template)
 ln0!(Int32((i-1)%nt),Int32((i-1)÷nt),nt,sh,args...,u)
 nothing
end
function lnwarp_kernel!(i,k,args,nt)
 lnwarp_site!(k,Int32((i-1)%nt),Int32((i-1)÷nt),nt,args...)
 nothing
end
function factory(c,algorithm,BT)
 grid=algorithm==0 ? Int(c.N) : cld(Int(c.N)*32,BT)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=BT,blocks=grid,shmem_size=algorithm==0 ? 128 : 0,sync=false)
 if algorithm==0
  u=Val(Int(c.C)%4==0 ? 4 : 1)
  ()->JACC.parallel_for(ln0_kernel!,spec,grid*BT,c.args0,u,Int32(BT),LN_TEMPLATE)
 else
  k=Val(algorithm)
  ()->JACC.parallel_for(lnwarp_kernel!,spec,grid*BT,k,c.argswarp,Int32(BT))
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
layernorm_study(factory)
