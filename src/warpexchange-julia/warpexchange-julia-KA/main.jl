using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel unsafe_indices=true function exchange_kernel(input,output,n,blocks)
 tid=Int32(@index(Local,Linear))-1;bid=Int32(@index(Group,Linear))-1;lane=tid%Int32(32);warp=tid÷Int32(32);shared=@localmem Int32 (1032,)

 full=cld(n,Int32(1024))*Int32(1024);base=bid*Int32(1024)
 while base<full
  for item in Int32(0):Int32(3)
   idx=base+tid*Int32(4)+item
   value=idx<n ? (@inbounds input[idx+Int32(1)]) : Int32(0)
   @inbounds shared[warp*Int32(129)+lane*Int32(4)+item+Int32(1)]=value
  end
 @synchronize
  for item in Int32(0):Int32(3)
   idx=base+tid*Int32(4)+item
   if idx<n
    @inbounds output[idx+Int32(1)]=shared[warp*Int32(129)+Int32(32)*item+lane+Int32(1)]
   end
  end
  base+=blocks*Int32(1024)
 end
end
function factory(input,output,n)
 k=exchange_kernel(CUDABackend(),256);blocks=cld(n,Int32(256))
 ()->k(input,output,n,blocks;ndrange=Int(blocks)*256)
end
exchange_study(factory)
