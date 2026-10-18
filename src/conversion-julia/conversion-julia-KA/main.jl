using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
@kernel function convert_kernel!(dst,src,n)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds dst[i]=cast_value(eltype(dst),src[i]);end
end
function launch(dst,src,n)
 threads=min(Int(n),256)
 convert_kernel!(BACKEND,threads)(dst,src,n;ndrange=cld(Int(n),threads)*threads)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
