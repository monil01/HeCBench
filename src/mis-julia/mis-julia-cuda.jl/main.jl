using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function init!(nodes::Int32,edges::Int32,idx,status)
 tid=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(256)+Int32(CUDA.threadIdx().x)-Int32(1)
 init_body!(nodes,edges,idx,status,tid,Int32(6144))
 return
end
function findmins!(nodes::Int32,idx,list,status)
 tid=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(256)+Int32(CUDA.threadIdx().x)-Int32(1)
 findmins_body!(nodes,idx,list,status,tid,Int32(6144))
 return
end
function launch(phase,nodes,edges,idx,list,status)
 if phase==1
  CUDA.@cuda threads=256 blocks=24 init!(nodes,edges,idx,status)
 else
  CUDA.@cuda threads=256 blocks=24 findmins!(nodes,idx,list,status)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
