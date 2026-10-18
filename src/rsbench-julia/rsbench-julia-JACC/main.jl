using CUDA, JACC
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
lookup_device(i,args...)=rs_store(i,args...)
function launch(args...)
 n=args[10]
 spec=JACC.launch_spec(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,n,lookup_device,args...)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
