using JACC
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","hash_math.jl"))
# Portableatomichelpers reusedfromtheverifiedKA candidate, marginalorderedworkflow.
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
@inline function insert_jacc!(i,tk,tv,keys,values,n)
 if i<=n;insert_record!(i,tk,tv,keys,values);end
 return
end
@inline function delete_jacc!(i,tk,tv,keys,n)
 if i<=n;delete_record!(i,tk,tv,keys);end
 return
end
@inline function collect_jacc!(i,tk,tv,ok,ov,count)
 if i<=ST_CAP;collect_record!(i,tk,tv,ok,ov,count);end
 return
end
function spec(n)
 backend=JACC.get_backend(:cuda)
 JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(Int(n),256),shmem_size=0,sync=false)
end
launch_insert!(tk,tv,keys,values,n)=JACC.parallel_for(insert_jacc!,spec(n),Int(n),tk,tv,keys,values,n)
launch_delete!(tk,tv,keys,n)=JACC.parallel_for(delete_jacc!,spec(n),Int(n),tk,tv,keys,n)
launch_collect!(tk,tv,ok,ov,count)=JACC.parallel_for(collect_jacc!,spec(ST_CAP),Int(ST_CAP),tk,tv,ok,ov,count)
study_main(launch_insert!,launch_delete!,launch_collect!,"jacc")
