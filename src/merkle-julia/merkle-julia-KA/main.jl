using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function phase0!(leaves,out,offset,mds,a1,a2)
 idx=Int32(@index(Global,Linear))-Int32(1)
 rescue_merge!(leaves,Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
end
@kernel function phase1!(out,offset,mds,a1,a2)
 idx=Int32(@index(Global,Linear))-Int32(1)
 rescue_merge!(out,Int32(8)*offset+Int32(8)*idx,out,Int32(4)*(offset+idx),mds,a1,a2)
end
function launch(leaves,out,n,mds,a1,a2)
 offset=Int32(n÷2)
 phase0!(BACKEND,32)(leaves,out,offset,mds,a1,a2;ndrange=Int(offset))
 offset>>=1
 while offset>0
  CUDA.synchronize(CUDA.context());threads=min(Int(offset),32)
  phase1!(BACKEND,threads)(out,offset,mds,a1,a2;ndrange=Int(offset))
  offset>>=1
 end
 CUDA.synchronize(CUDA.context())
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
