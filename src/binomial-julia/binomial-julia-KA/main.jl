using CUDA
using KernelAbstractions
include("study_kernel.jl")
const BACKEND=CUDA.CUDABackend()
function launch(n,args...)
 binomial_kernel!(BACKEND,128)(args...;ndrange=128n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
