using CUDA
@inline block_barrier()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(A,B,b,input,finished,head)
 s=CuStaticSharedArray(Int32,2);cycle_body!(Int32(threadIdx().x)-Int32(1),A,B,b,input,finished,head,s);return
end
run!(A,B,b,input,finished,head,tmp)=(@cuda threads=64 blocks=16 kernel!(A,B,b,input,finished,head);nothing)
study(run!)
