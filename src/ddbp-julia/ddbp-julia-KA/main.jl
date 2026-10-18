using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function stage_kernel!(A,p,v,stage)
 index=@index(Global,NTuple)
 I=(Int64(index[1]-1),Int64(index[2]-1),Int64(index[3]-1))
 ddbp_lane!(I,A,p,v,stage)
end
function launch(stage,grid,threads,A,p,v)
 k=stage_kernel!(CUDABackend(),threads)
 dims=(grid[1]*threads[1],grid[2]*threads[2],grid[3]*threads[3])
 k(A,p,v,stage;ndrange=dims)
end
ddbp_study(launch)
