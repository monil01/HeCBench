include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA,KernelAbstractions
@kernel function remote_double_ka!(source,target)
 i=@index(Global,Linear)
 @inbounds target[i+1]=source[i+1]*2f0
end
allocate(n)=KernelAbstractions.allocate(CUDA.CUDABackend(),Float32,(n,))
function launch(n,source,target)
 remote_double_ka!(KernelAbstractions.get_backend(target),256)(source,target;ndrange=n)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
