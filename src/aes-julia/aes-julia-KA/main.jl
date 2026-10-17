using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(output,input,key,table,template,::Val{INV}) where INV
 gx,gy=@index(Group,NTuple);lx,ly=@index(Local,NTuple);bx=Int32(gx);by=Int32(gy);row=Int32(ly);shared=@localmem NTuple{4,UInt8} (8,)
 index=((by-1)*16+(bx-1))*4+row
 r0=row-1
 @inbounds shared[row]=xorword(input[index],key[(INV ? 40 : 0)+row])
 if INV
  for round in Int32(9):-Int32(1):Int32(1)
   @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table)
   @synchronize
   @inbounds shared[4+row]=xorword(shared[row],key[round*4+row])
   @synchronize
   @inbounds shared[row]=mixword(shared,Int32(4),r0,Val(true))
  end
  @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table);output[index]=xorword(shared[row],key[row])
 else
  for round in Int32(1):Int32(9)
   @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false))
   @synchronize
   @inbounds shared[4+row]=mixword(shared,Int32(0),r0,Val(false))
   @synchronize
   @inbounds shared[row]=xorword(shared[4+row],key[round*4+row])
  end
  @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false));output[index]=xorword(shared[row],key[40+row])
 end
end
run!(args...)=kernel!(get_backend(args[1]),(1,4))(args...;ndrange=(16,64))
study(run!)
