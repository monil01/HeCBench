using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function atomic_kernel!(a,loops)
 i=@index(Global,Linear)
 system_lane!(a,Int32(i-1),loops)
end
function factory(a,loops)
 k=atomic_kernel!(CUDABackend(),8);()->k(a,loops;ndrange=8)
end
atomic_study(factory)
