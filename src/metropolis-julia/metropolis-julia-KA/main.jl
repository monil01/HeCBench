using CUDA,KernelAbstractions
include("../metropolis_device.jl")
@kernel unsafe_indices=true function metropolis_kernel!(S,H,A,I,h,B,alt)
 t=@index(Local,NTuple);g=@index(Group,NTuple)
 tx=Int(t[1])-1;ty=Int(t[2])-1;tz=Int(t[3])-1;bx=Int(g[1])-1;by=Int(g[2])-1;bz=Int(g[3])-1
 ss=@localmem Int32 (1080,)
 c,st,inc=tile_load!(ss,S,H,A,I,tx,ty,tz,bx,by,bz,alt);@synchronize
 st=metropolis_phase!(ss,c,h,B,st,inc,Val(0));@synchronize
 st=metropolis_phase!(ss,c,h,B,st,inc,Val(1));@synchronize
 tile_store!(ss,S,A,I,c,st,inc)
end
const MK=metropolis_kernel!(CUDABackend(),(16,4,4))
function launch(c)
 for step in 1:c.ns,alt in 0:1
  MK(c.vs...,c.h,c.B,alt;ndrange=(32,8,32));CUDA.synchronize()
 end
 nothing
end
include("../study_common.jl")
metropolis_study(launch)
