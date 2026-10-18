using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function neuron_kernel!(args...)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 point!(i,args...)
 return
end
function launch(h,m,n,ge,v,gi,last,nr,N,dt,t,P)
 @cuda threads=256 blocks=cld(Int(N),256) neuron_kernel!(h,m,n,ge,v,gi,last,nr,N,dt,t,P)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
