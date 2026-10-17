using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function map_jacc!(i,pages,rank,maps,degrees,n)
 body_map!(Int32(i)-Int32(1),pages,rank,maps,degrees,n)
 nothing
end
@inline function reduce_jacc!(j,rank,maps,n,history)
 body_reduce!(Int32(j)-Int32(1),rank,maps,n,history)
 nothing
end
const SPEC=JACC.launch_spec(threads=256,blocks=2,shmem_size=0,sync=false)
function run!(pages,rank,maps,degrees,n,history)
 JACC.parallel_for(SPEC,512,map_jacc!,pages,rank,maps,degrees,n)
 JACC.parallel_for(SPEC,512,reduce_jacc!,rank,maps,n,history)
 nothing
end
study(run!)
