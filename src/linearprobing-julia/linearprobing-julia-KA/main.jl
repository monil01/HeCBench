using KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","hash_math.jl"))
@inline function insert_record!(i,tk,tv,keys,values)
 @inbounds key=keys[i]
 @inbounds value=values[i]
 slot=portable_hash(key)
 while true
  idx=slot+UInt32(1)
  prev=(Atomix.@atomicreplace :monotonic tk[idx] ST_EMPTY=>key).old
  if prev==ST_EMPTY||prev==key
   Atomix.@atomicswap :monotonic tv[idx]=value
   return
  end
  slot=(slot+UInt32(1))&(ST_CAP-UInt32(1))
 end
end
@inline function delete_record!(i,tk,tv,keys)
 @inbounds key=keys[i]
 slot=portable_hash(key)
 while true
  idx=slot+UInt32(1)
  @inbounds stored=tk[idx]
  if stored==key
   @inbounds tv[idx]=ST_EMPTY
   return
  end
  stored==ST_EMPTY&&return
  slot=(slot+UInt32(1))&(ST_CAP-UInt32(1))
 end
end
@inline function collect_record!(i,tk,tv,ok,ov,count)
 @inbounds key=tk[i]
 if key!=ST_EMPTY
  @inbounds value=tv[i]
  if value!=ST_EMPTY
   pos=(Atomix.@atomic :monotonic count[1]+=UInt32(1))
   @inbounds ok[pos]=key
   @inbounds ov[pos]=value
  end
 end
 return
end
@kernel function insert_ka!(tk,tv,keys,values,n)
 i=@index(Global,Linear)
 if i<=n;insert_record!(i,tk,tv,keys,values);end
end
@kernel function delete_ka!(tk,tv,keys,n)
 i=@index(Global,Linear)
 if i<=n;delete_record!(i,tk,tv,keys);end
end
@kernel function collect_ka!(tk,tv,ok,ov,count)
 i=@index(Global,Linear)
 if i<=ST_CAP;collect_record!(i,tk,tv,ok,ov,count);end
end
launch_insert!(tk,tv,keys,values,n)=insert_ka!(KernelAbstractions.get_backend(tk),256)(tk,tv,keys,values,n;ndrange=Int(n))
launch_delete!(tk,tv,keys,n)=delete_ka!(KernelAbstractions.get_backend(tk),256)(tk,tv,keys,n;ndrange=Int(n))
launch_collect!(tk,tv,ok,ov,count)=collect_ka!(KernelAbstractions.get_backend(tk),256)(tk,tv,ok,ov,count;ndrange=Int(ST_CAP))
study_main(launch_insert!,launch_delete!,launch_collect!,"ka")
