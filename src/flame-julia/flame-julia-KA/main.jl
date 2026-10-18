using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","flame_math.jl"))
@kernel unsafe_indices=true function initialize_kernel!(args...)
 idx=Int32(@index(Global,Linear)-1)
 initialize_site!(idx,args...)
end
@kernel unsafe_indices=true function iterate_kernel!(args...)
 idx=Int32(@index(Global,Linear)-1)
 iterate_site!(idx,args...)
end
@kernel unsafe_indices=true function generate_kernel!(args...)
 idx=Int32(@index(Global,Linear)-1)
 generate_site!(idx,args...)
end
function launch_initialize(model,args...)
 initialize_kernel!(CUDABackend(),32)(args...;ndrange=256)
end
function launch_iterate(model,args...)
 iterate_kernel!(CUDABackend(),32)(args...;ndrange=256)
end
function launch_generate(model,args...)
 generate_kernel!(CUDABackend(),32)(args...;ndrange=256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:ka)
