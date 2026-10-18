using CUDA,KernelAbstractions,Atomix
include("../study_portable_common.jl")
@kernel unsafe_indices=true function ga_kernel(n,data,best,winner,rng,bfsm,same,smax,sbest,old,states)
 lid=Int32(@index(Local,Linear));bid=Int32(@index(Group,Linear));gid=(bid-1)*Int32(256)+lid
 next=@localmem UInt8 (4096,)

 if lid==1
  @inbounds old[bid]=0;same[bid]=0
 end
 @synchronize
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
  @synchronize
  Atomix.@atomic max(smax[bid],hits)
  @synchronize
  if hits==(@inbounds smax[bid])
   Atomix.@atomic max(sbest[bid],Int32(lid-1))
  end
  @synchronize
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
  @synchronize
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
end
@kernel unsafe_indices=true function max_kernel(best,bfsm)
 @inbounds bid=best[1]
 for k in 1:16
  @inbounds best[k+3]=Int32(bfsm[16*bid+k])
 end
end
function factory(n,data,best,winner,rng,bfsm,same,smax,sbest,old,states)
 ga=ga_kernel(CUDABackend(),256);finish=max_kernel(CUDABackend(),1)
 ()->begin
  ga(n,data,best,winner,rng,bfsm,same,smax,sbest,old,states;ndrange=262144)
  finish(best,bfsm;ndrange=1)
 end
end
fsm_study(factory)
