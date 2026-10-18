using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function phase0!(leaves,out,offset,mds,a1,a2)
 idx=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 rescue_merge!(leaves,Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
 return
end
function phase1!(out,offset,mds,a1,a2)
 idx=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 rescue_merge!(out,Int32(8)*offset+Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
 return
end
function launch(leaves,out,n,mds,a1,a2)
 offset=Int32(n÷2)
 CUDA.@cuda threads=32 blocks=Int(offset)÷32 phase0!(leaves,out,offset,mds,a1,a2)
 offset>>=1
 while offset>0
  CUDA.synchronize(CUDA.context());threads=min(Int(offset),32)
  CUDA.@cuda threads=threads blocks=Int(offset)÷threads phase1!(out,offset,mds,a1,a2)
  offset>>=1
 end
 CUDA.synchronize(CUDA.context())
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
