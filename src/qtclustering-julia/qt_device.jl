# Full compact-storage source algorithm. CUDA shared-memory/barrier backend
# extensions are explicit for all Julia dispatch models; no CPU clustering path.
@inline function closest_qt(min_dist,threshold,point,tid)
 ds=CUDA.CuStaticSharedArray(Float32,64);ps=CUDA.CuStaticSharedArray(Int32,64)
 @inbounds ds[tid+1]=min_dist;@inbounds ps[tid+1]=point
 CUDA.sync_threads()
 if tid==0
  for j in 2:64
   @inbounds d=ds[j]
   @inbounds if d<min_dist || (d==min_dist && ps[j]<ps[1]);min_dist=d;ps[1]=ps[j];end
  end
  if min_dist>threshold;@inbounds ps[1]=Int32(-1);end
 end
 CUDA.sync_threads()
 @inbounds ps[1]
end
@inline function evaluate_qt(cand,r,seed,latest,last,min_dist,point,x,ao,co,N,D,threshold,tid)
 if cand==seed;return cand,last,min_dist,point;end
 k=r*64+tid
 @inbounds curr=x.cache[co+k+1]
 @inbounds if curr>threshold || x.ai[ao+cand+1]!=0 || x.clustered[cand+1]!=0
  return Int32(seed),last,min_dist,point
 end
 newdist=threshold+1f0
 for j in last:D-1
  @inbounds q=x.idx[latest*D+j+1]
  if q>cand || q<0;last=j;break;end
  if q==cand;@inbounds newdist=x.dist[latest*D+j+1];break;end
 end
 diameter=max(curr,newdist)
 if newdist>curr;@inbounds x.cache[co+k+1]=diameter;end
 if diameter<min_dist;min_dist=diameter;point=cand;end
 cand,last,min_dist,point
end
@inline function candidate_qt(seed,x,ao,co,pc,N,D,threshold,tid,::Val{EMIT}) where EMIT
 for i in tid:64:N-1;@inbounds x.ai[ao+i+1]=Int8(0);end
 for i in tid:64:D-1;@inbounds x.cache[co+i+1]=0f0;end
 if tid==0
  @inbounds x.ai[ao+seed+1]=Int8(1)
  if EMIT;@inbounds x.result[1]=Int32(seed);end
 end
 CUDA.sync_threads()
 cands=ntuple(_->Int32(-1),Val(12))
 for r in 0:11
  k=r*64+tid
  if k>=D;break;end
  @inbounds q=x.idx[seed*D+k+1]
  cands=Base.setindex(cands,q,r+1)
  if q<0;break;end
 end
 CUDA.sync_threads()
 count=1;latest=seed;flag=true
 while count<pc && flag
  min_dist=3f0*threshold;point=Int32(-1);last=0
  for r in 0:11
   cand=cands[r+1]
   if cand<0;break;end
   cand,last,min_dist,point=evaluate_qt(cand,r,seed,latest,last,min_dist,point,x,ao,co,N,D,threshold,tid)
   cands=Base.setindex(cands,cand,r+1)
  end
  CUDA.sync_threads()
  r=12
  while r*64+tid<D
   @inbounds cand=x.idx[seed*D+r*64+tid+1]
   if cand<0;break;end
   cand,last,min_dist,point=evaluate_qt(cand,r,seed,latest,last,min_dist,point,x,ao,co,N,D,threshold,tid)
   r+=1
  end
  CUDA.sync_threads()
  chosen=closest_qt(min_dist,threshold,point,tid)
  if chosen>=0
   if tid==0
    @inbounds x.ai[ao+chosen+1]=Int8(1)
    if EMIT;@inbounds x.result[count+1]=chosen;end
   end
   latest=Int(chosen);count+=1
  else
   flag=false
  end
  CUDA.sync_threads()
 end
 CUDA.sync_threads()
 count
end
@inline function qt_site!(::Val{:degree},x,pc,N,D,threshold,blocks,seed)
 tid=Int(CUDA.threadIdx().x)-1;bid=Int(CUDA.blockIdx().x)-1
 localcount=cld(N,blocks);start=bid*localcount;count=min(localcount,max(0,N-start))
 for i in tid:64:count-1
  n=0
  for j in 0:D-1;@inbounds n+=x.idx[(start+i)*D+j+1]>=0;end
  @inbounds x.degree[start+i+1]=Int32(n)
 end
 nothing
end
@inline function qt_site!(::Val{:qtc},x,pc,N,D,threshold,blocks,seed)
 tid=Int(CUDA.threadIdx().x)-1;bid=Int(CUDA.blockIdx().x)-1
 best=-1;winner=Int32(-12345)
 for i in bid:blocks:pc-1
  @inbounds s=Int(x.ungrp[i+1]);@inbounds degree=x.degree[s+1]
  if degree<=best;continue;end
  count=candidate_qt(s,x,bid*N,bid*D,pc,N,D,threshold,tid,Val(false))
  if count>best;best=count;winner=Int32(s);end
 end
 if tid==0;@inbounds x.card[2*bid+1]=Int32(best);@inbounds x.card[2*bid+2]=winner;end
 nothing
end
@inline function qt_site!(::Val{:reduce},x,pc,N,D,threshold,blocks,seed)
 best=Int32(-1);winner=Int32(-12345)
 for i in 0:blocks-1
  @inbounds if x.card[2*i+1]>best;best=x.card[2*i+1];winner=x.card[2*i+2];end
 end
 @inbounds x.card[1]=best;@inbounds x.card[2]=winner
 nothing
end
@inline function qt_site!(::Val{:trim},x,pc,N,D,threshold,blocks,seed)
 tid=Int(CUDA.threadIdx().x)-1
 candidate_qt(seed,x,0,0,pc,N,D,threshold,tid,Val(true))
 tmp=CUDA.CuStaticSharedArray(Int32,64);cnt=CUDA.CuStaticSharedArray(Int32,1);flag=CUDA.CuStaticSharedArray(Bool,1)
 if tid==0;@inbounds cnt[1]=Int32(0);@inbounds flag[1]=false;end
 CUDA.sync_threads()
 # Uniform masked tail and a lifetime barrier before in-place writes protect
 # source shared reads. Native source remains unchanged; extra Julia barrier is
 # an explicit synchronization/mapping covariate, not an observed native defect.
 for i in 0:64:pc-1
  active=i+tid<pc
  if active;@inbounds tmp[tid+1]=x.ungrp[i+tid+1];end
  CUDA.sync_threads()
  if active
   @inbounds point=tmp[tid+1]
   @inbounds if x.ai[point+1]==1
    flag[1]=true;tmp[tid+1]=Int32(-42)
   else
    x.ungrp[Int(cnt[1])+tid+1]=point
   end
  end
  CUDA.sync_threads()
  if tid==0
   @inbounds count=Int(cnt[1])
   @inbounds if flag[1]
    for j in 0:min(63,pc-i-1)
     if tmp[j+1]!=-42;x.ungrp[count+1]=tmp[j+1];count+=1;end
    end
    cnt[1]=Int32(count)
   else
    cnt[1]+=Int32(64)
   end
   @inbounds flag[1]=false
  end
  CUDA.sync_threads()
 end
 nothing
end
@inline function qt_site!(::Val{:update},x,pc,N,D,threshold,blocks,seed)
 tid=Int(CUDA.threadIdx().x)-1
 for i in tid:64:N-1;@inbounds x.clustered[i+1] |= x.ai[i+1];end
 CUDA.sync_threads()
 nothing
end
