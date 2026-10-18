using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","dslash_math.jl"))
@kernel unsafe_indices=true function back_kernel!(args...)
 i=@index(Global,Linear)
 back_site!(i-1,args...)
end
@kernel unsafe_indices=true function hopping_kernel!(args...)
 i=@index(Global,Linear)
 dslash_site!(i-1,args...)
end
function launch_back(model,args...)
 back_kernel!(CUDABackend(),1)(args...;ndrange=Int(args[end]))
end
function launch_dslash(model,args...)
 hopping_kernel!(CUDABackend(),256)(args...;ndrange=cld(Int(args[end]),256)*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:ka)
