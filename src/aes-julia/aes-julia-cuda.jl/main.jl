using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(output,input,key,table,template,::Val{INV}) where INV
 bx=Int32(blockIdx().x);by=Int32(blockIdx().y);row=Int32(threadIdx().y);shared=CuStaticSharedArray(NTuple{4,UInt8},8)
 index=((by-1)*16+(bx-1))*4+row
 r0=row-1
 @inbounds shared[row]=xorword(input[index],key[(INV ? 40 : 0)+row])
 if INV
  for round in Int32(9):-Int32(1):Int32(1)
   @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table)
   sync_threads()
   @inbounds shared[4+row]=xorword(shared[row],key[round*4+row])
   sync_threads()
   @inbounds shared[row]=mixword(shared,Int32(4),r0,Val(true))
  end
  @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table);output[index]=xorword(shared[row],key[row])
 else
  for round in Int32(1):Int32(9)
   @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false))
   sync_threads()
   @inbounds shared[4+row]=mixword(shared,Int32(0),r0,Val(false))
   sync_threads()
   @inbounds shared[row]=xorword(shared[4+row],key[round*4+row])
  end
  @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false));output[index]=xorword(shared[row],key[40+row])
 end
 return
end
run!(args...)=(@cuda threads=(1,4) blocks=(16,16) kernel!(args...))
study(run!)
