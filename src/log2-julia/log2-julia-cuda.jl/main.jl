using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function log_cuda!(input,out,n::Int32,row::Int32,p::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;@inbounds out[1+row*n+i]=binary_log_study(input[i],p);end
 return
end
function factory(input,out,n)
 (row,p)->(@cuda threads=256 blocks=cld(n,256) log_cuda!(input,out,n,row,p))
end
log_study(factory)
