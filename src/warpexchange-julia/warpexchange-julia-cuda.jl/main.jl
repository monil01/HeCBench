using CUDA
include("../study_common.jl")
function exchange_kernel(input,output,n)
 tid=Int32(threadIdx().x)-1;bid=Int32(blockIdx().x)-1;blocks=Int32(gridDim().x);lane=tid%Int32(32);warp=tid÷Int32(32);shared=CuStaticSharedArray(Int32,1032)

 full=cld(n,Int32(1024))*Int32(1024);base=bid*Int32(1024)
 while base<full
  for item in Int32(0):Int32(3)
   idx=base+tid*Int32(4)+item
   value=idx<n ? (@inbounds input[idx+Int32(1)]) : Int32(0)
   @inbounds shared[warp*Int32(129)+lane*Int32(4)+item+Int32(1)]=value
  end
 sync_warp()
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
factory(input,output,n)=()->(@cuda threads=256 blocks=cld(n,Int32(256)) exchange_kernel(input,output,n))
exchange_study(factory)
