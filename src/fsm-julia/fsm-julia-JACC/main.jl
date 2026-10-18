using JACC,CUDA,Atomix
include("../study_portable_common.jl")
@inline function ga_kernel(i,n,data,best,winner,rng,bfsm,same,smax,sbest,old,states,template)
 gid=Int32(i);lid=(gid-Int32(1))%Int32(256)+Int32(1);bid=(gid-Int32(1))÷Int32(256)+Int32(1)
 next=JACC.shared(template)

 if lid==1
  @inbounds old[bid]=0;same[bid]=0
 end
 JACC.sync_workgroup()
 @inbounds rng[gid]=UInt32(1234) ⊻ UInt32(gid-1)
 fsm_random!(rng,gid)
 for k in 1:16
  @inbounds next[16*(lid-1)+k]=UInt8(fsm_random!(rng,gid)&UInt32(7))
 end
 while true
  # Packed four-byte state words remain private to each logical lane.
  for word in 0:8191
   @inbounds states[word*262144+gid]=UInt32(0)
  end
  hits=Int32(0)
  for i in 1:n
   @inbounds d=Int32(data[i]);pc=d>>1;bit=d&Int32(1)
   idx=Int64(pc>>2)*262144+Int64(gid);shift=(pc&Int32(3))*Int32(8)
   @inbounds word=states[idx];s=Int32((word>>shift)&UInt32(255))
   hits+=Int32((s&Int32(1))==bit)
   @inbounds newstate=UInt32(next[16*(lid-1)+2*s+bit+1]);states[idx]=(word&~(UInt32(255)<<shift))|(newstate<<shift)
  end
  if lid==1
   Atomix.@atomic best[3]+=Int32(1)
   @inbounds smax[bid]=0;sbest[bid]=0
  end
  JACC.sync_workgroup()
  Atomix.@atomic max(smax[bid],hits)
  JACC.sync_workgroup()
  if hits==(@inbounds smax[bid])
   Atomix.@atomic max(sbest[bid],Int32(lid-1))
  end
  JACC.sync_workgroup()
  @inbounds selected=sbest[bid]+1
  mutate=false
  if selected==lid
   @inbounds same[bid]+=UInt8(1)
   if (@inbounds old[bid])<(@inbounds smax[bid])
    @inbounds old[bid]=smax[bid];same[bid]=0
   end
  else
   mutate=(fsm_random!(rng,gid)&UInt32(7))==UInt32(0)
  end
  JACC.sync_workgroup()
  for k in 1:16
   a=fsm_random!(rng,gid);b=fsm_random!(rng,gid);rnd=a&b
   if selected!=lid
    @inbounds v=UInt32(next[16*(selected-1)+k]);own=UInt32(next[16*(lid-1)+k]);next[16*(lid-1)+k]=UInt8(mutate ? ((v ⊻ rnd)&UInt32(7)) : ((own&rnd)|(v&~rnd)))
   end
  end
  if (@inbounds same[bid])>=UInt8(1)
   if selected==lid
    key=(UInt64(hits)<<32)+UInt64(bid-1)
    Atomix.@atomic max(winner[1],key)
    for k in 1:16
     @inbounds bfsm[16*(bid-1)+k]=next[16*(lid-1)+k]
    end
   end
   break
  end
 end
 return
end
@inline function max_kernel(i,best,bfsm)
 @inbounds bid=best[1]
 for k in 1:16
  @inbounds best[k+3]=Int32(bfsm[16*bid+k])
 end
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(n,data,best,winner,rng,bfsm,same,smax,sbest,old,states)
 template=CUDA.zeros(UInt8,4096)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=1024,shmem_size=4096,sync=false)
 finish=JACC.LaunchSpec{typeof(BACKEND)}(;threads=1,blocks=1,sync=false)
 ()->begin
  JACC.parallel_for(spec,262144,ga_kernel,n,data,best,winner,rng,bfsm,same,smax,sbest,old,states,template)
  JACC.parallel_for(finish,1,max_kernel,best,bfsm)
 end
end
fsm_study(factory)
