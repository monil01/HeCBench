using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
function relax_kernel!(data,idx,cols,u,f,n::Int32)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 amg_relax(i,n,data,idx,cols,u,f)
 nothing
end
function observe_kernel!(p,n::Int32)
 @inbounds p[1]=n;p[2]=Int32(NX);p[3]=Int32(NY);p[4]=Int32(NZ);p[5]=Int32(R)
 return
end
amg_study((args...)->CUDA.@cuda(threads=256,blocks=cld(N,256),relax_kernel!(args...)),(p,n)->CUDA.@cuda(threads=1,blocks=1,observe_kernel!(p,n)),"cuda_jl")
