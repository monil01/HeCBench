using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","rainflow_math.jl"))
function rainflow_kernel!(index,history,offsets,extrema,points,results,lengths,n)
    id=Int32(index)
    if id<=n;rainflow_eval!(id,history,offsets,extrema,points,results,lengths);end
    nothing
end
function study_prepare(history,offsets,extrema,points,results,lengths,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->JACC.parallel_for(rainflow_kernel!,spec,n,history,offsets,extrema,points,results,lengths,Int32(n))
end
exit(study_main(study_prepare))
