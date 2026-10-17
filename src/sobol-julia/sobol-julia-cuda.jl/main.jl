include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run!(dirs,out,n,d,gx,template)
 @cuda threads=128 blocks=(gx,d) sobol_kernel!(dirs,out,n,d)
 nothing
end
study(run!)
