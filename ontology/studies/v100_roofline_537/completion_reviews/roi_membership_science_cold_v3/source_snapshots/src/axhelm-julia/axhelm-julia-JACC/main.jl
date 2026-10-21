ENV["JULIA_CUDA_MEMORY_POOL"]="none"
using CUDA
import JACC
JACC.@init_backend
@inline block_barrier()=JACC.sync_workgroup()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function scalar!(x,y,offset,G,D,L,Q,A,tmp)
 ix=Int32(x)-Int32(1);s=JACC.shared(tmp);helm_body!(ix%Int32(8),Int32(y)-Int32(1),ix÷Int32(8),offset,G,D,L,Q,A,s,Val(1));return
end
function fused3!(x,y,offset,G,D,L,Q,A,tmp)
 ix=Int32(x)-Int32(1);s=JACC.shared(tmp);helm_body!(ix%Int32(8),Int32(y)-Int32(1),ix÷Int32(8),offset,G,D,L,Q,A,s,Val(3));return
end
function run!(fields,offset,G,D,L,Q,A)
 size=fields==1 ? 256 : 640;f=fields==1 ? scalar! : fused3!
 tmp=view(G,1:size)
 JACC.parallel_for(JACC.launch_spec(threads=(8,8),blocks=(STUDY_ELEMENTS,1),shmem_size=4size,sync=false),(8STUDY_ELEMENTS,8),f,offset,G,D,L,Q,A,tmp)
 return
end
study(run!)
