using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
@kernel function relax_kernel!(data,idx,cols,u,f,n::Int32)
 i=Int32(@index(Global,Linear))-Int32(1)
 amg_relax(i,n,data,idx,cols,u,f)
end
@kernel function observe_kernel!(p,n::Int32)
 @inbounds p[1]=n;p[2]=Int32(NX);p[3]=Int32(NY);p[4]=Int32(NZ);p[5]=Int32(R)
end
const RELAX=relax_kernel!(CUDABackend(),256)
const OBSERVE=observe_kernel!(CUDABackend(),1)
amg_study((args...)->RELAX(args...;ndrange=cld(N,256)*256),(p,n)->OBSERVE(p,n;ndrange=1),"ka")
