using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function map_ka!(pages,rank,maps,degrees,n)
 i0=Int32(@index(Global,Linear))-Int32(1)
 body_map!(i0,pages,rank,maps,degrees,n)
end
@kernel function reduce_ka!(rank,maps,n,history)
 j0=Int32(@index(Global,Linear))-Int32(1)
 body_reduce!(j0,rank,maps,n,history)
end
const BACKEND=CUDABackend()
function run!(pages,rank,maps,degrees,n,history)
 map_ka!(BACKEND,256)(pages,rank,maps,degrees,n;ndrange=512)
 reduce_ka!(BACKEND,256)(rank,maps,n,history;ndrange=512)
 nothing
end
study(run!)
