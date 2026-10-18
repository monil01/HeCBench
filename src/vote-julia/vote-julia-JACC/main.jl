using CUDA,JACC
JACC.@init_backend
function vote_jacc!(index,input,out,repeat::Int32,::Val{MODE}) where MODE
 tx=Int32(index)
 for r in Int32(1):repeat
  @inbounds pred=input[tx]!=UInt32(0)
  v=MODE==0 ? CUDA.vote_any_sync(UInt32(0xffffffff),pred) : CUDA.vote_all_sync(UInt32(0xffffffff),pred)
  @inbounds out[tx]=UInt32(v)
 end
 return
end
function info_jacc!(index,out,warp_size::Int32,repeat::Int32)
 tx=Int32(index)-Int32(1);offset=Int32(3)*tx;pred=tx>=warp_size*Int32(3)÷Int32(2)
 for r in Int32(1):repeat
  a=CUDA.vote_any_sync(UInt32(0xffffffff),pred)
  @inbounds out[offset+Int32(1)]=UInt8(a)
  @inbounds out[offset+Int32(2)]=UInt8(pred)
  if CUDA.vote_all_sync(UInt32(0xffffffff),pred);@inbounds out[offset+Int32(3)]=UInt8(1);end
 end
 return
end
function factory(input,out,n,op,repeat)
 spec=JACC.launch_spec(;threads=n,blocks=1,shmem_size=0,sync=false)
 if op==2;return ()->JACC.parallel_for(spec,n,info_jacc!,out,Int32(32),Int32(repeat));end
 mode=Val(op);()->JACC.parallel_for(spec,n,vote_jacc!,input,out,Int32(repeat),mode)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
vote_study(factory)
