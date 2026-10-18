using CUDA
include("../study_device.jl")
include("../study_common.jl")
function lut_kernel(W,A,B,X,Y,M,K,nb,G,::Val{T},::Val{BIAS}) where {T,BIAS}
 t=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1
 lut=CuStaticSharedArray(Float16,T÷8*256)
 lut_build!(lut,X,t,by,Val(T));sync_threads();lut_finish!(lut,W,A,B,X,Y,M,K,nb,G,t,bx,by,Val(T),Val(BIAS));nothing
end
function launch(c)
 @cuda threads=256 blocks=(cld(c.M,2048),c.K÷c.T) lut_kernel(c.vs...,c.y,c.M,c.K,c.nb,c.G,Val(c.T),Val(c.bi==1))
end
lut_study(launch)
