using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function init!(i,nodes::Int32,edges::Int32,idx,status)
 tid=Int32(i)-Int32(1)
 init_body!(nodes,edges,idx,status,tid,Int32(6144))
 return nothing
end
function findmins!(i,nodes::Int32,idx,list,status)
 tid=Int32(i)-Int32(1)
 findmins_body!(nodes,idx,list,status,tid,Int32(6144))
 return nothing
end
function launch(phase,nodes,edges,idx,list,status)
 spec=JACC.launch_spec(;threads=256,blocks=24,shmem_size=0,sync=false)
 if phase==1
  JACC.parallel_for(spec,6144,init!,nodes,edges,idx,status)
 else
  JACC.parallel_for(spec,6144,findmins!,nodes,idx,list,status)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
