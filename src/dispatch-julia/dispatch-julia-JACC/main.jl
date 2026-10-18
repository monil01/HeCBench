using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
empty_kernel(i)=nothing
const specification=JACC.launch_spec(;threads=1,blocks=1,shmem_size=0,sync=false)
launch()=JACC.parallel_for(specification,1,empty_kernel)
dispatch_study(launch)
