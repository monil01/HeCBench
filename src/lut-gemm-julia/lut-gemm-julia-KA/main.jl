using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel unsafe_indices=true function lut_kernel!(W,A,B,X,Y,M,K,nb,G,::Val{T},::Val{BIAS}) where {T,BIAS}
 t=Int(@index(Local,Linear))-1;g=@index(Group,NTuple);bx=Int(g[1])-1;by=Int(g[2])-1
 lut=@localmem Float16 (T÷8*256,)
 lut_build!(lut,X,t,by,Val(T));@synchronize;lut_finish!(lut,W,A,B,X,Y,M,K,nb,G,t,bx,by,Val(T),Val(BIAS))
end
const LK=lut_kernel!(CUDABackend(),(256,1))
function launch(c)
 LK(c.vs...,c.y,c.M,c.K,c.nb,c.G,Val(c.T),Val(c.bi==1);ndrange=(cld(c.M,2048)*256,c.K÷c.T));nothing
end
include("../study_common.jl")
lut_study(launch)
