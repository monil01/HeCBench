using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","layernorm_math.jl"))
@kernel unsafe_indices=true function ln0_kernel!(args,u,nt)
 sh=@localmem Float32 (32,)
 ln0!(Int32(@index(Local,Linear)-1),Int32(@index(Group,Linear)-1),nt,sh,args...,u)
end
@kernel unsafe_indices=true function lnwarp_kernel!(k,args,nt)
 lnwarp_site!(k,Int32(@index(Local,Linear)-1),Int32(@index(Group,Linear)-1),nt,args...)
end
function factory(c,algorithm,BT)
 if algorithm==0
  u=Val(Int(c.C)%4==0 ? 4 : 1);k=ln0_kernel!(CUDABackend(),BT)
  ()->k(c.args0,u,Int32(BT);ndrange=BT*Int(c.N))
 else
  v=Val(algorithm);k=lnwarp_kernel!(CUDABackend(),BT);grid=cld(Int(c.N)*32,BT)
  ()->k(v,c.argswarp,Int32(BT);ndrange=grid*BT)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
layernorm_study(factory)
