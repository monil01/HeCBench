using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function iv_cuda!(src,dst,n::Int32,layout)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;iv_accumulate!(src,dst,i,layout);end
 return
end
factory(src,dst)=(n,layout)->(@cuda threads=256 blocks=cld(n,256) iv_cuda!(src,dst,n,layout))
interleave_study(factory)
