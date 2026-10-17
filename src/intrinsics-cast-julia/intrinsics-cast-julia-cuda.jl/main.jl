using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function cast_from_kernel!(input,out,n::Int32)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=n;@inbounds out[i]=cast_from_fp(input[i]);end
 return
end
function cast_to_kernel!(input,out,n::Int32)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=n;@inbounds out[i]=cast_to_fp(input[i]);end
 return
end
function factory(input,out,n,mode)
 k=mode=="from_fp" ? cast_from_kernel! : cast_to_kernel!
 return ()->(@cuda threads=256 blocks=cld(n,256) k(input,out,Int32(n)))
end
if abspath(PROGRAM_FILE)==@__FILE__;cast_study(factory);end
