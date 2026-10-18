using CUDA
include(joinpath(@__DIR__,"..","dslash_math.jl"))
function back_kernel!(args...)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 back_site!(i,args...);return
end
function hopping_kernel!(args...)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 dslash_site!(i,args...);return
end
function launch_back(model,args...)
 @cuda threads=1 blocks=Int(args[end]) back_kernel!(args...)
end
function launch_dslash(model,args...)
 @cuda threads=256 blocks=cld(Int(args[end]),256) hopping_kernel!(args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main(:cuda_jl)
