using CUDA
include("../study_device.jl")
include("../study_common.jl")
function stage_kernel!(A,p,v,stage)
 I=(Int64((blockIdx().x-1)*blockDim().x+threadIdx().x-1),Int64((blockIdx().y-1)*blockDim().y+threadIdx().y-1),Int64((blockIdx().z-1)*blockDim().z+threadIdx().z-1))
 ddbp_lane!(I,A,p,v,stage)
 nothing
end
launch(stage,grid,threads,A,p,v)=CUDA.@cuda(threads=threads,blocks=grid,stage_kernel!(A,p,v,stage))
ddbp_study(launch)
