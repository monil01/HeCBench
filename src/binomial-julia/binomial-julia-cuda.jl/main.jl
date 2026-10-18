using CUDA
include("study_kernel.jl")
function launch(n,args...)
 @cuda threads=128 blocks=n binomial_kernel!(args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
