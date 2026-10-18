using CUDA,JACC
include("../xs_device.jl")
function lookup_kernel!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 if i<=N;xprimary!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
 nothing
end
function observe_kernel!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M,N)
 if i<=N;xobserve!(i,out,num,concs,mats,grid,union,idx,I,G,typ,bins,M);end
 nothing
end
function launch(c)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=c.blocks,shmem_size=0,sync=false)
 JACC.parallel_for(lookup_kernel!,spec,c.blocks*256,c.out.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N)
 nothing
end
function observe(c)
 spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=c.blocks,shmem_size=0,sync=false)
 JACC.parallel_for(observe_kernel!,spec,c.blocks*256,c.obs.data,c.num.data,c.concs.data,c.mats.data,c.grid.data,c.energy.data,c.idx.data,c.I,c.G,c.typ,c.bins,c.M,c.N)
 nothing
end
include("../study_common.jl")
xsbench_study(launch,observe)
