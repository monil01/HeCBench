using CUDA
include(joinpath(@__DIR__,"..","study_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,vol,col,pars)
 if k==1;CUDA.@cuda threads=512 blocks=1 vol2col_kernel!(Int64(210),vol,pars...,col)
 else;CUDA.@cuda threads=512 blocks=1 col2vol_kernel!(Int64(210),col,pars...,vol)
 end
end
study((T,n)->CUDA.zeros(T,n),launch)
