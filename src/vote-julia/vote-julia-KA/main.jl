using CUDA,KernelAbstractions
@kernel function vote_ka!(input,out,repeat::Int32,::Val{MODE}) where MODE
 tx=Int32(@index(Local,Linear))
 for r in Int32(1):repeat
  @inbounds pred=input[tx]!=UInt32(0)
  v=MODE==0 ? CUDA.vote_any_sync(UInt32(0xffffffff),pred) : CUDA.vote_all_sync(UInt32(0xffffffff),pred)
  @inbounds out[tx]=UInt32(v)
 end
end
@kernel function info_ka!(out,warp_size::Int32,repeat::Int32)
 tx=Int32(@index(Local,Linear))-Int32(1);offset=Int32(3)*tx;pred=tx>=warp_size*Int32(3)÷Int32(2)
 for r in Int32(1):repeat
  a=CUDA.vote_any_sync(UInt32(0xffffffff),pred)
  @inbounds out[offset+Int32(1)]=UInt8(a)
  @inbounds out[offset+Int32(2)]=UInt8(pred)
  if CUDA.vote_all_sync(UInt32(0xffffffff),pred);@inbounds out[offset+Int32(3)]=UInt8(1);end
 end
end
function factory(input,out,n,op,repeat)
 if op==2
  k=info_ka!(CUDABackend(),n)
  return ()->k(out,Int32(32),Int32(repeat);ndrange=n)
 end
 mode=Val(op);k=vote_ka!(CUDABackend(),n)
 ()->k(input,out,Int32(repeat),mode;ndrange=n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
vote_study(factory)
