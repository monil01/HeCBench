using CUDA
include(joinpath(@__DIR__,"..","study_constants.jl"))
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
launch(M,n,g,f,v,a,c,t)=CUDA.@cuda threads=128 blocks=256 spm_kernel!(M,Int32(n),g,f,Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),a,c,t)
study((T,n)->CUDA.zeros(T,n),launch)
