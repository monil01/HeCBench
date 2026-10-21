ENV["JULIA_CUDA_MEMORY_POOL"]="none"
using CUDA
@inline block_barrier()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function scalar!(offset,G,D,L,Q,A)
 s=CuStaticSharedArray(Float32,256);helm_body!(Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockIdx().x)-Int32(1),offset,G,D,L,Q,A,s,Val(1));return
end
function fused3!(offset,G,D,L,Q,A)
 s=CuStaticSharedArray(Float32,640);helm_body!(Int32(threadIdx().x)-Int32(1),Int32(threadIdx().y)-Int32(1),Int32(blockIdx().x)-Int32(1),offset,G,D,L,Q,A,s,Val(3));return
end
function run!(fields,offset,G,D,L,Q,A)
 if fields==1;@cuda threads=(8,8) blocks=STUDY_ELEMENTS scalar!(offset,G,D,L,Q,A)
 else;@cuda threads=(8,8) blocks=STUDY_ELEMENTS fused3!(offset,G,D,L,Q,A)
 end
 return
end
study(run!)
