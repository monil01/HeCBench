# Source-derived ECL-CC device stages, adapted zero-based CSR and explicit device counters.
# Original license/source provenance retained in src/cc-cuda/main.cu.
@inline function cc_representative(idx::Int32,stat)
 @inbounds curr=stat[idx+Int32(1)]
 if curr!=idx
  prev=idx
  @inbounds next=stat[curr+Int32(1)]
  while curr>next
   @inbounds stat[prev+Int32(1)]=next
   prev=curr;curr=next
   @inbounds next=stat[curr+Int32(1)]
  end
 end
 curr
end
@inline function cc_union(vstat::Int32,ostat::Int32,stat)
 while vstat!=ostat
  if vstat<ostat
   ret=CUDA.atomic_cas!(pointer(stat,Int(ostat)+1),ostat,vstat)
   ret==ostat&&break
   ostat=ret
  else
   ret=CUDA.atomic_cas!(pointer(stat,Int(vstat)+1),vstat,ostat)
   ret==vstat&&break
   vstat=ret
  end
 end
 vstat
end
@inline function cc_init(from::Int32,incr::Int32,n::Int32,idx,adj,stat,wl,ctr)
 v=from
 while v<n
  @inbounds beg=idx[v+Int32(1)];en=idx[v+Int32(2)]
  m=v;i=beg
  while m==v&&i<en
   @inbounds m=min(m,adj[i+Int32(1)])
   i+=Int32(1)
  end
  @inbounds stat[v+Int32(1)]=m
  v+=incr
 end
 if from==0
  @inbounds ctr[1]=0;ctr[2]=0;ctr[3]=n-Int32(1);ctr[4]=n-Int32(1)
 end
 nothing
end
@inline function cc_compute1(from::Int32,incr::Int32,n::Int32,idx,adj,stat,wl,ctr)
 v=from
 while v<n
  @inbounds vstat=stat[v+Int32(1)]
  if v!=vstat
   @inbounds beg=idx[v+Int32(1)];en=idx[v+Int32(2)]
   deg=en-beg
   if deg>16
    q=deg<=352 ? CUDA.atomic_add!(pointer(ctr,1),Int32(1)) : CUDA.atomic_add!(pointer(ctr,3),Int32(-1))
    @inbounds wl[q+Int32(1)]=v
   else
    vstat=cc_representative(v,stat)
    for i in beg:en-Int32(1)
     @inbounds other=adj[i+Int32(1)]
     if v>other;vstat=cc_union(vstat,cc_representative(other,stat),stat);end
    end
   end
  end
  v+=incr
 end
 nothing
end
@inline function cc_compute2(from::Int32,incr::Int32,n::Int32,idx,adj,stat,wl,ctr)
 lane=from%Int32(32);q=Int32(0)
 if lane==0;q=CUDA.atomic_add!(pointer(ctr,2),Int32(1));end
 q=CUDA.shfl_sync(typemax(UInt32),q,Int32(0))
 @inbounds top=ctr[1]
 while q<top
  @inbounds v=wl[q+Int32(1)];beg=idx[v+Int32(1)];en=idx[v+Int32(2)]
  vstat=cc_representative(v,stat)
  for i in beg+lane:Int32(32):en-Int32(1)
   @inbounds other=adj[i+Int32(1)]
   if v>other;vstat=cc_union(vstat,cc_representative(other,stat),stat);end
  end
  if lane==0;q=CUDA.atomic_add!(pointer(ctr,2),Int32(1));end
  q=CUDA.shfl_sync(typemax(UInt32),q,Int32(0))
 end
 nothing
end
@inline function cc_compute3(from::Int32,incr::Int32,n::Int32,idx,adj,stat,wl,ctr)
 tid=from%Int32(256);shared=CUDA.CuStaticSharedArray(Int32,1)
 if tid==0
  q=CUDA.atomic_add!(pointer(ctr,4),Int32(-1))
  @inbounds shared[1]=q>ctr[3] ? wl[q+Int32(1)] : Int32(-1)
 end
 CUDA.sync_threads()
 @inbounds v=shared[1]
 while v>=0
  CUDA.sync_threads()
  vstat=cc_representative(v,stat)
  @inbounds beg=idx[v+Int32(1)];en=idx[v+Int32(2)]
  for i in beg+tid:Int32(256):en-Int32(1)
   @inbounds other=adj[i+Int32(1)]
   if v>other;vstat=cc_union(vstat,cc_representative(other,stat),stat);end
  end
  if tid==0
   q=CUDA.atomic_add!(pointer(ctr,4),Int32(-1))
   @inbounds shared[1]=q>ctr[3] ? wl[q+Int32(1)] : Int32(-1)
  end
  CUDA.sync_threads()
  @inbounds v=shared[1]
 end
 nothing
end
@inline function cc_flatten(from::Int32,incr::Int32,n::Int32,idx,adj,stat,wl,ctr)
 v=from
 while v<n
  @inbounds vstat=stat[v+Int32(1)]
  old=vstat
  @inbounds next=stat[vstat+Int32(1)]
  while vstat>next
   vstat=next
   @inbounds next=stat[vstat+Int32(1)]
  end
  if old!=vstat;@inbounds stat[v+Int32(1)]=vstat;end
  v+=incr
 end
 nothing
end
