using CUDA
function convert_kernel!(dst,src,n)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=n;@inbounds dst[i]=cast_value(eltype(dst),src[i]);end
 return
end
function launch(dst,src,n)
 threads=min(Int(n),256)
 @cuda threads=threads blocks=cld(Int(n),threads) convert_kernel!(dst,src,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
