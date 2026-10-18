include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
doh_study((a,h,w,sigma,o,n)->(@cuda threads=256 blocks=cld(n,256) hessian_matrix_det_kernel!(a,h,w,sigma,o)))
