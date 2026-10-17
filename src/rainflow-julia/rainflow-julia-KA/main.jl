using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","rainflow_math.jl"))
@kernel function rainflow_kernel!(history,offsets,extrema,points,results,lengths,n)
    id=Int32(@index(Global,Linear))
    if id<=n;rainflow_eval!(id,history,offsets,extrema,points,results,lengths);end
    nothing
end
function study_prepare(history,offsets,extrema,points,results,lengths,n)
    k=rainflow_kernel!(CUDABackend(),256)
    ()->k(history,offsets,extrema,points,results,lengths,Int32(n);ndrange=n)
end
exit(study_main(study_prepare))
