include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA, KernelAbstractions
CUDA.allowscalar(false)
@kernel function reduce_device!(out,input,n)
 tid=@index(Global,Linear)
 pos=Int(tid);total=0f0
 while pos<=n
  @inbounds total+=input[pos]
  pos+=8192
 end
 @inbounds out[tid]=total
end
function launch(o,i,n)
 reduce_device!(CUDABackend(),256)(o,i,Int32(n);ndrange=8192)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->KernelAbstractions.allocate(CUDABackend(),Float32,(n,)),launch)
