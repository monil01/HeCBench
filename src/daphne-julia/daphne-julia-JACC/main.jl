using CUDA,JACC
include(joinpath(@__DIR__,"..","study_math.jl"))
function daphne_jacc!(i,cp,x,y,z,v,n)
    project_item!(i,cp,x,y,z,v,n)
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(cp,x,y,z,v,n)
    JACC.parallel_for(SPEC,Int(n),daphne_jacc!,cp,x,y,z,v,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"));daphne_study(launch_jacc!)
