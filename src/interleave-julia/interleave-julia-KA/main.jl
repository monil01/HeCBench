using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function iv_ka!(src,dst,n::Int32,layout)
 i=Int32(@index(Global,Linear))
 if i<=n;iv_accumulate!(src,dst,i,layout);end
end
function factory(src,dst)
 k=iv_ka!(CUDABackend(),256)
 (n,layout)->k(src,dst,n,layout;ndrange=n)
end
interleave_study(factory)
