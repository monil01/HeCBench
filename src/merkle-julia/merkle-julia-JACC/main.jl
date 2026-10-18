using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function phase0!(i,leaves,out,offset,mds,a1,a2)
 idx=Int32(i)-Int32(1)
 rescue_merge!(leaves,Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
 return nothing
end
function phase1!(i,out,offset,mds,a1,a2)
 idx=Int32(i)-Int32(1)
 rescue_merge!(out,Int32(8)*offset+Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
 return nothing
end
function launch(leaves,out,n,mds,a1,a2)
 offset=Int32(n÷2)
 spec=JACC.launch_spec(;threads=32,blocks=Int(offset)÷32,shmem_size=0,sync=false)
 JACC.parallel_for(spec,Int(offset),phase0!,leaves,out,offset,mds,a1,a2)
 offset>>=1
 while offset>0
  CUDA.synchronize(CUDA.context());threads=min(Int(offset),32)
  spec=JACC.launch_spec(;threads=threads,blocks=Int(offset)÷threads,shmem_size=0,sync=false)
  JACC.parallel_for(spec,Int(offset),phase1!,out,offset,mds,a1,a2)
  offset>>=1
 end
 CUDA.synchronize(CUDA.context())
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
