using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_copy!(dst,src,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds dst[idx]=src[idx]
    end
    nothing
end
function study_prepare(dst,src,n)
    kernel=portable_copy!(CUDABackend(),256);()->kernel(dst,src,Int32(n);ndrange=cld(n,256)*256)
end
exit(study_main(study_prepare))
