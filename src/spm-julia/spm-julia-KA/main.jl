using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_constants.jl"))
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
launch(M,n,g,f,v,a,c,t)=spm_ka!(BACKEND,128)(M,Int32(n),g,f,Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),a,c,t;ndrange=128*256)
study((T,n)->KernelAbstractions.zeros(BACKEND,T,n),launch)
