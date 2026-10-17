using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fresnel_kernel!(input,output,n)
    i=Int32(@index(Global,Linear))
    if i<=n
        @inbounds output[i]=fresnel_corrected(input[i])
    end
    nothing
end
function study_prepare(input,output,n)
    k=fresnel_kernel!(CUDABackend(),256)
    ()->k(input,output,Int32(n);ndrange=n)
end
exit(study_main(study_prepare))
