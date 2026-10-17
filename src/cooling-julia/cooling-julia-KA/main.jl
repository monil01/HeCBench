using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function cooling_kernel!(temp,result,n,heat)
    i=@index(Global,Linear)
    if i<=n
        @inbounds result[i]=primordial_cool(.0899,temp[i],heat)
    end
end
function study_prepare(temp,result,n,heat)
    k=cooling_kernel!(CUDABackend(),256)
    ()->k(temp,result,Int32(n),heat;ndrange=n)
end
exit(study_main(study_prepare))
