using CUDA
include("../xs_device.jl")
function lookup_kernel!(out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=N;xprimary!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
 return
end
function observe_kernel!(out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=N;xobserve!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
 return
end
function launch(c)
 @cuda threads=256 blocks=c.blocks lookup_kernel!(c.out.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N)
 nothing
end
function observe(c)
 @cuda threads=256 blocks=c.blocks observe_kernel!(c.obs.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N)
 nothing
end
include("../study_common.jl")
xsbench_study(launch,observe)
