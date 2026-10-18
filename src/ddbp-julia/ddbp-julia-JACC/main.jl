using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function stage_kernel!(i,j,k,A,p,v,stage)
 I=(Int64(i-1),Int64(j-1),Int64(k-1))
 ddbp_lane!(I,A,p,v,stage)
 nothing
end
function launch(stage,grid,threads,A,p,v)
 spec=JACC.launch_spec(;threads=threads,blocks=grid,shmem_size=0,sync=false)
 dims=(grid[1]*threads[1],grid[2]*threads[2],grid[3]*threads[3])
 JACC.parallel_for(spec,dims,stage_kernel!,A,p,v,stage)
end
ddbp_study(launch)
