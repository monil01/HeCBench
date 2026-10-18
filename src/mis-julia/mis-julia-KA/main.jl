using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function init!(nodes::Int32,edges::Int32,idx,status)
 tid=Int32(@index(Global,Linear))-Int32(1)
 init_body!(nodes,edges,idx,status,tid,Int32(6144))
end
@kernel function findmins!(nodes::Int32,idx,list,status)
 tid=Int32(@index(Global,Linear))-Int32(1)
 findmins_body!(nodes,idx,list,status,tid,Int32(6144))
end
function launch(phase,nodes,edges,idx,list,status)
 if phase==1
  init!(BACKEND,256)(nodes,edges,idx,status;ndrange=6144)
 else
  findmins!(BACKEND,256)(nodes,idx,list,status;ndrange=6144)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
