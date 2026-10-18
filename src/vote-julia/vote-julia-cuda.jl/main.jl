using CUDA
function vote_jl!(input,out,repeat::Int32,::Val{MODE}) where MODE
 tx=threadIdx().x
 for r in Int32(1):repeat
  @inbounds pred=input[tx]!=UInt32(0)
  v=MODE==0 ? CUDA.vote_any_sync(UInt32(0xffffffff),pred) : CUDA.vote_all_sync(UInt32(0xffffffff),pred)
  @inbounds out[tx]=UInt32(v)
 end
 return
end
function info_jl!(out,warp_size::Int32,repeat::Int32)
 tx=threadIdx().x-Int32(1);offset=Int32(3)*tx;pred=tx>=warp_size*Int32(3)÷Int32(2)
 for r in Int32(1):repeat
  a=CUDA.vote_any_sync(UInt32(0xffffffff),pred)
  @inbounds out[offset+Int32(1)]=UInt8(a)
  @inbounds out[offset+Int32(2)]=UInt8(pred)
  if CUDA.vote_all_sync(UInt32(0xffffffff),pred);@inbounds out[offset+Int32(3)]=UInt8(1);end
 end
 return
end
function factory(input,out,n,op,repeat)
 if op==2
  k=@cuda launch=false info_jl!(out,Int32(32),Int32(repeat))
  return ()->k(out,Int32(32),Int32(repeat);threads=n,blocks=1)
 end
 mode=Val(op);k=@cuda launch=false vote_jl!(input,out,Int32(repeat),mode)
 ()->k(input,out,Int32(repeat),mode;threads=n,blocks=1)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
vote_study(factory)
