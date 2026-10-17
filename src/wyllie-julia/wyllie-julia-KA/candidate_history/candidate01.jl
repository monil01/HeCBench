using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function wyllie_ka!(list,n)
 index=@index(Global,Linear)
 if index<=n
  @inbounds while true
   node=list[index];(node>>32)==-1&&break
   next=list[(node>>32)+1];(next>>32)==-1&&break
   temp=(node&Int64(0xffffffff))+(next&Int64(0xffffffff))+((next>>32)<<32)
   @synchronize
   list[index]=temp
  end
 end
end
launch!(d,n)=wyllie_ka!(KernelAbstractions.get_backend(d),256)(d,n;ndrange=Int(n))
study_main(launch!)
