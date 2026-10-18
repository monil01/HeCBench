using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function che_kernel!(stage,args...)
 x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);z=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)
 point!(x,y,z,stage,args...)
 return
end
function launch(stage,cold,cnew,mu,ff,nx,ny,nz)
 @cuda threads=(16,4,4) blocks=(cld(nx,16),cld(ny,4),cld(nz,4)) che_kernel!(Val(stage),cold,cnew,mu,ff,nx,ny,nz)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
