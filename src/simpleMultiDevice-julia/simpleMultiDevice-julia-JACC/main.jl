include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA
import JACC
JACC.@init_backend
CUDA.allowscalar(false)
@inline function reduce_device!(tid,out,input,n)
 pos=Int(tid);total=0f0
 while pos<=n
  @inbounds total+=input[pos]
  pos+=8192
 end
 @inbounds out[tid]=total
 nothing
end
function launch(o,i,n)
 spec=JACC.launch_spec(;threads=256,blocks=32,shmem_size=0,sync=false)
 JACC.parallel_for(spec,8192,reduce_device!,o,i,Int32(n))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->JACC.zeros(Float32,n),launch)
