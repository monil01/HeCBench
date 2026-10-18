using CUDA
include("../study_device.jl")
include("../study_common.jl")
function atomic_kernel!(a,loops)
 tid=Int32(threadIdx().x-1);system_lane!(a,tid,loops);nothing
end
factory(a,loops)=()->CUDA.@cuda threads=8 blocks=1 atomic_kernel!(a,loops)
atomic_study(factory)
