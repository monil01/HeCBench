using CUDA, KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function lookup_device(args...)
 i=@index(Global,Linear)
 rs_store(i,args...)
end
function launch(args...)
 lookup_device(KernelAbstractions.get_backend(args[8]),256)(args...;ndrange=256cld(args[10],256))
 nothing
end
function allocate(v)
 a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),size(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
