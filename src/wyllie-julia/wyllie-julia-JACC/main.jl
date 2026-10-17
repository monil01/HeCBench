using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function wyllie_jacc!(index,list,n)
 if index<=n
  @inbounds while true
   node=list[index];(node>>32)==-1&&break
   next=list[(node>>32)+1];(next>>32)==-1&&break
   temp=(node&Int64(0xffffffff))+(next&Int64(0xffffffff))+((next>>32)<<32)
   JACC.sync_workgroup()
   list[index]=temp
  end
 end
 return
end
function launch!(d,n)
 backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
 JACC.parallel_for(wyllie_jacc!,spec,Int(n),d,n)
end
study_main(launch!)
