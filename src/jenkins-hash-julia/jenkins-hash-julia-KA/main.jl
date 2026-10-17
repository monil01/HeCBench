using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","jenkins_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function jenkins_kernel!(lengths,initvals,keys,out,n::Int64)
    id=Int64(@index(Global,Linear))
    if id<=n
        idx=Int(id)
        @inbounds out[idx]=hashlittle_words(keys,(idx-1)*16+1,lengths[idx],initvals[idx])
    end
    nothing
end
function study_prepare(lens,seeds,keys,out,n)
    k=jenkins_kernel!(CUDABackend(),256)
    ()->k(lens,seeds,keys,out,Int64(n);ndrange=n)
end
exit(study_main(study_prepare))
