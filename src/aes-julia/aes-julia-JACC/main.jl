using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(gx,gy,output,input,key,table,template,::Val{INV}) where INV
 bx=Int32(gx);by=Int32(div(gy-1,4)+1);row=Int32(mod(gy-1,4)+1);shared=JACC.shared(template)
 index=((by-1)*16+(bx-1))*4+row
 r0=row-1
 @inbounds shared[row]=xorword(input[index],key[(INV ? 40 : 0)+row])
 if INV
  for round in Int32(9):-Int32(1):Int32(1)
   @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table)
   JACC.sync_workgroup()
   @inbounds shared[4+row]=xorword(shared[row],key[round*4+row])
   JACC.sync_workgroup()
   @inbounds shared[row]=mixword(shared,Int32(4),r0,Val(true))
  end
  @inbounds shared[row]=rotateword(shared[row],r0,Val(true));shared[row]=subword(shared[row],table);output[index]=xorword(shared[row],key[row])
 else
  for round in Int32(1):Int32(9)
   @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false))
   JACC.sync_workgroup()
   @inbounds shared[4+row]=mixword(shared,Int32(0),r0,Val(false))
   JACC.sync_workgroup()
   @inbounds shared[row]=xorword(shared[4+row],key[round*4+row])
  end
  @inbounds shared[row]=subword(shared[row],table);shared[row]=rotateword(shared[row],r0,Val(false));output[index]=xorword(shared[row],key[40+row])
 end
 return
end
run!(args...)=JACC.parallel_for(JACC.launch_spec(threads=(1,4),blocks=(16,16),shmem_size=32,sync=false),(16,64),kernel!,args...)
study(run!)
