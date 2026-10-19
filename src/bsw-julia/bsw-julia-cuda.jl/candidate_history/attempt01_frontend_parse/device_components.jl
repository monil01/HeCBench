# Source-faithful BSW anti-diagonal device algorithm. CUDA.jl owns these intrinsics;
# KA/JACC entrypoints using this component are CUDA-specific device hybrids.
@inline bsw_short(x::Int32)=x % Int16
@inline function bsw_warp_reduce(val::Int16,ind::Int16,ind2::Int16,lengthB::Int32,inverse::Bool)
 tid=Int32(CUDA.threadIdx().x)-Int32(1)
 mask=CUDA.vote_ballot_sync(typemax(UInt32),tid<lengthB);mymax=val
 for offset in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  temp=bsw_short(CUDA.shfl_down_sync(mask,Int32(val),offset));val=max(val,temp)
  ni=bsw_short(CUDA.shfl_down_sync(mask,Int32(ind),offset));nj=bsw_short(CUDA.shfl_down_sync(mask,Int32(ind2),offset))
  if val!=mymax
   ind=ni;ind2=nj;mymax=val
  elseif val==temp
   if (inverse&&nj>ind2)||(!inverse&&ni<ind)
    ind=ni;ind2=nj
   end
  end
 end
 (mymax,ind,ind2)
end
@inline function bsw_block_reduce(val::Int16,ind::Int16,ind2::Int16,lengthB::Int32,inverse::Bool)
 tid=Int32(CUDA.threadIdx().x)-Int32(1);lane=tid%Int32(32);warp=tid÷Int32(32)
 totals=CUDA.CuStaticSharedArray(Int16,32);inds=CUDA.CuStaticSharedArray(Int16,32);inds2=CUDA.CuStaticSharedArray(Int16,32)
 val,mi,mj=bsw_warp_reduce(val,ind,ind2,lengthB,inverse)
 if lane==0
  @inbounds totals[warp+1]=val;inds[warp+1]=mi;inds2[warp+1]=mj
 end
 CUDA.sync_threads();nwarps=(Int32(32)+Int32(CUDA.blockDim().x)-Int32(1))÷Int32(32)
 if tid<nwarps
  @inbounds val=totals[tid+1];mi=inds[tid+1];mj=inds2[tid+1]
 else
  val=Int16(0);mi=Int16(-1);mj=Int16(-1)
 end
 CUDA.sync_threads()
 if warp==0
  val,ind,ind2=bsw_warp_reduce(val,mi,mj,lengthB,inverse)
 end
 (val,ind,ind2)
end
@inline function bsw_device!(inverse::Bool,a,b,offsetA,offsetB,beginA,endA,beginB,endB,scores,openGap::Int16,extendGap::Int16,scoring,encoding)
 tid=Int32(CUDA.threadIdx().x)-Int32(1);bid=Int32(CUDA.blockIdx().x)-Int32(1);lane=tid%Int32(32);warp=tid÷Int32(32)
 @inbounds startA=bid==0 ? Int32(0) : Int32(offsetA[bid]);startB=bid==0 ? Int32(0) : Int32(offsetB[bid]);lengthA=Int32(offsetA[bid+1])-startA;lengthB=Int32(offsetB[bid+1])-startB
 if inverse
  @inbounds lengthA=Int32(endA[bid+1]);lengthB=Int32(endB[bid+1])
 end
 maxsize=max(lengthA,lengthB);minsize=min(lengthA,lengthB)
 valid=CUDA.CuDynamicSharedArray(Int8,SHMEM)
 p=tid
 while p<minsize
  @inbounds valid[p+1]=0;valid[minsize+p+1]=1;valid[2minsize+p+1]=0
  p+=Int32(32)
 end
 base=2minsize;char=UInt8(0)
 if lengthA<lengthB
  if tid<lengthA
   @inbounds char=a[startA+(inverse ? lengthA-tid : tid+1)]
  end
 else
  if tid<lengthB
   @inbounds char=b[startB+(inverse ? lengthB-tid : tid+1)]
  end
 end
 CUDA.sync_threads();i=Int32(1);threadmax=Int16(0);maxi=Int16(0);maxj=Int16(0)
 h=Int16(0);f=Int16(0);e=Int16(0);ph=Int16(0);pf=Int16(0);pe=Int16(0);pph=Int16(0);ppf=Int16(0);ppe=Int16(0)
 shE=CUDA.CuStaticSharedArray(Int16,32);shH=CUDA.CuStaticSharedArray(Int16,32);shHH=CUDA.CuStaticSharedArray(Int16,32)
 spillE=CUDA.CuStaticSharedArray(Int16,1024);spillH=CUDA.CuStaticSharedArray(Int16,1024);spillHH=CUDA.CuStaticSharedArray(Int16,1024)
 sc=CUDA.CuStaticSharedArray(Int16,576);en=CUDA.CuStaticSharedArray(Int16,91)
 p=tid;step=Int32(CUDA.blockDim().x)
 while p<576
  @inbounds sc[p+1]=scoring[p+1];p+=step
 end
 p=tid
 while p<91
  @inbounds en[p+1]=encoding[p+1];p+=step
 end
 CUDA.sync_threads()
 for diag in Int32(0):(lengthA+lengthB-Int32(2))
  base-=Int32(diag<minsize||diag>=maxsize)
  pph,ph,h=ph,h,Int16(0);ppe,pe,e=pe,e,Int16(0);ppf,pf,f=pf,f,Int16(0)
  if lane==31
   @inbounds shE[warp+1]=pe;shH[warp+1]=ph;shHH[warp+1]=pph
  end
  if diag>=maxsize
   @inbounds spillE[tid+1]=pe;spillH[tid+1]=ph;spillHH[tid+1]=pph
  end
  CUDA.sync_threads()
  @inbounds isvalid=valid[base+tid+1]!=0
  if isvalid&&tid<minsize
   mask=CUDA.vote_ballot_sync(CUDA.active_mask(),isvalid&&tid<minsize)
   fv=pf+extendGap;hf=ph+openGap
   # CUDA.jl indexed shuffle lanes are 1-based; native lane-1 wraps modulo32.
   source_lane=mod(lane-Int32(1),Int32(32))+Int32(1)
   es=bsw_short(CUDA.shfl_sync(mask,Int32(pe),source_lane));hs=bsw_short(CUDA.shfl_sync(mask,Int32(ph),source_lane))
   if diag>=maxsize
    @inbounds ev=spillE[tid]+extendGap;he=spillH[tid]+openGap
   elseif warp!=0&&lane==0
    @inbounds ev=shE[warp]+extendGap;he=shH[warp]+openGap
   else
    ev=es+extendGap;he=hs+openGap
   end
   if warp==0&&lane==0;ev=Int16(0);he=Int16(0);end
   f=max(fv,hf);e=max(ev,he)
   hh=bsw_short(CUDA.shfl_sync(mask,Int32(pph),source_lane))
   if diag>=maxsize
    @inbounds hh=spillHH[tid]
   elseif warp!=0&&lane==0
    @inbounds hh=shHH[warp]
   end
   if warp==0&&lane==0;hh=Int16(0);end
   if lengthA<lengthB
    @inbounds other=b[startB+(inverse ? maxsize-i+1 : i)]
   else
    @inbounds other=a[startA+(inverse ? maxsize-i+1 : i)]
   end
   @inbounds q=Int32(en[Int32(other)+1]);rr=Int32(en[Int32(char)+1]);add=sc[q*Int32(24)+rr+1]
   h=max(hh+add,f,e,Int16(0));ii=inverse ? maxsize-i : i;jj=inverse ? minsize-tid-Int32(1) : tid+Int32(1)
   if threadmax<h
    maxi=Int16(ii);maxj=Int16(jj);threadmax=h
   end
   i+=Int32(1)
  end
  CUDA.sync_threads()
 end
 threadmax,maxi,maxj=bsw_block_reduce(threadmax,maxi,maxj,minsize,inverse)
 if tid==0
  if inverse
   if lengthA<lengthB
    @inbounds beginB[bid+1]=maxi;beginA[bid+1]=maxj
   else
    @inbounds beginA[bid+1]=maxi;beginB[bid+1]=maxj
   end
  else
   if lengthA<lengthB
    @inbounds endB[bid+1]=maxi;endA[bid+1]=maxj;scores[bid+1]=threadmax
   else
    @inbounds endA[bid+1]=maxi;endB[bid+1]=maxj;scores[bid+1]=threadmax
   end
  end
 end
 nothing
end
