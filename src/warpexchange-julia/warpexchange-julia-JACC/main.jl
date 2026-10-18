using CUDA,JACC
include("../study_portable_common.jl")
@inline function exchange_kernel(i,input,output,n,blocks,template)
 tid=Int32((i-1)%256);bid=Int32((i-1)÷256);lane=tid%Int32(32);warp=tid÷Int32(32);shared=JACC.shared(template)

 full=cld(n,Int32(1024))*Int32(1024);base=bid*Int32(1024)
 while base<full
  for item in Int32(0):Int32(3)
   idx=base+tid*Int32(4)+item
   value=idx<n ? (@inbounds input[idx+Int32(1)]) : Int32(0)
   @inbounds shared[warp*Int32(129)+lane*Int32(4)+item+Int32(1)]=value
  end
 JACC.sync_workgroup()
  for item in Int32(0):Int32(3)
   idx=base+tid*Int32(4)+item
   if idx<n
    @inbounds output[idx+Int32(1)]=shared[warp*Int32(129)+Int32(32)*item+lane+Int32(1)]
   end
  end
  base+=blocks*Int32(1024)
 end
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(input,output,n)
 blocks=cld(n,Int32(256));template=CUDA.zeros(Int32,1032)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=Int(blocks),shmem_size=4128,sync=false)
 ()->JACC.parallel_for(spec,Int(blocks)*256,exchange_kernel,input,output,n,blocks,template)
end
exchange_study(factory)
