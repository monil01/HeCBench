using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function atomic_kernel!(i,a,loops)
 system_lane!(a,Int32(i-1),loops)
 nothing
end
function factory(a,loops)
 spec=JACC.launch_spec(;threads=8,blocks=1,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,8,atomic_kernel!,a,loops)
end
atomic_study(factory)
