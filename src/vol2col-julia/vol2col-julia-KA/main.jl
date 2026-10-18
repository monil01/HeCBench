using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_cpu_reference.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(k,vol,col,pars)
 if k==1;vol2col_ka!(BACKEND,512)(Int64(210),vol,pars...,col,Int64(512);ndrange=512)
 else;col2vol_ka!(BACKEND,512)(Int64(210),col,pars...,vol,Int64(512);ndrange=512)
 end
end
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
