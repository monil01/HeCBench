using CUDA,KernelAbstractions
include("../xs_device.jl")
@kernel unsafe_indices=true function lookup_kernel!(out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 i=@index(Global,Linear)
 if i<=N;xprimary!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
end
@kernel unsafe_indices=true function observe_kernel!(out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 i=@index(Global,Linear)
 if i<=N;xobserve!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
end
const primary=lookup_kernel!(CUDABackend(),256);const diagnostic=observe_kernel!(CUDABackend(),256)
function launch(c)
 primary(c.out.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N;ndrange=c.blocks*256)
 nothing
end
function observe(c)
 diagnostic(c.obs.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N;ndrange=c.blocks*256)
 nothing
end
include("../study_common.jl")
xsbench_study(launch,observe)
