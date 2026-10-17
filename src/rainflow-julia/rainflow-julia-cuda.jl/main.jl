using CUDA
include(joinpath(@__DIR__,"..","rainflow_math.jl"))
function rainflow_kernel!(history,offsets,extrema,points,results,lengths,n)
    id=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
    if id<=n;rainflow_eval!(id,history,offsets,extrema,points,results,lengths);end
    nothing
end
function study_prepare(history,offsets,extrema,points,results,lengths,n)
    ()->(@cuda threads=256 blocks=cld(n,256) rainflow_kernel!(history,offsets,extrema,points,results,lengths,Int32(n)))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
