using CUDA
import JACC
JACC.@init_backend
@inline block_barrier()=JACC.sync_workgroup()
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,A,B,b,input,finished,head,tmp)
 s=JACC.shared(tmp);cycle_body!((Int32(i)-Int32(1))%Int32(64),A,B,b,input,finished,head,s);return
end
run!(A,B,b,input,finished,head,tmp)=(JACC.parallel_for(JACC.launch_spec(threads=64,blocks=16,shmem_size=8,sync=false),1024,kernel!,A,B,b,input,finished,head,tmp);nothing)
study(run!)
