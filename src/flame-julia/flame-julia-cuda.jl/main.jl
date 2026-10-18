using CUDA
include(joinpath(@__DIR__,"..","flame_math.jl"))
function initialize_kernel!(args...)
 idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 initialize_site!(idx,args...);return
end
function iterate_kernel!(args...)
 idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 iterate_site!(idx,args...);return
end
function generate_kernel!(args...)
 idx=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 generate_site!(idx,args...);return
end
function launch_initialize(model,args...)
 @cuda threads=32 blocks=8 initialize_kernel!(args...)
end
function launch_iterate(model,args...)
 @cuda threads=32 blocks=8 iterate_kernel!(args...)
end
function launch_generate(model,args...)
 @cuda threads=32 blocks=8 generate_kernel!(args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:cuda_jl)
