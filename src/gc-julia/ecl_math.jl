# Native ECL-GC mathematical shortcut/bitset updates, mapping adaptation:
# sequential neighbor traversal per vertex replaces warp compaction/reductions.
# Redistribution terms from Texas State University retained in LICENSE.
import Atomix
const ECL_MSB=UInt32(0x80000000)
@inline function ecl_hash(v::UInt32)
 v=((v>>16)⊻v)*UInt32(0x45d9f3b);v=((v>>16)⊻v)*UInt32(0x45d9f3b)
 (v>>16)⊻v
end
@inline function ecl_priority(v,nei,degv,degn)
 hv=ecl_hash(UInt32(v));hn=ecl_hash(UInt32(nei))
 degv<degn||(degv==degn&&(hv<hn||(hv==hn&&v<nei)))
end
@inline function ecl_init!(v,n,e,idx,list,list2,pc,pc2,col,wl,sz)
 if v<n
  @inbounds beg=Int(idx[v+1]);stop=Int(idx[v+2]);degree=stop-beg;range=0;active=UInt32(0)
  for k in beg:stop-1
   @inbounds nei=Int(list[k+1]);degn=Int(idx[nei+2]-idx[nei+1])
   if ecl_priority(v,nei,degree,degn)
    if degree>=32
     @inbounds list2[beg+range+1]=Int32(nei)
    else
     active|=ECL_MSB>>(k-beg)
    end
    range+=1
   end
  end
  @inbounds col[v+1]=(degree>=32||range==0) ? UInt32(range)<<16 : active
  # Source signed arithmetic MSB >> range means all leading range+1 bits.
  @inbounds pc[v+1]=range>=32 ? typemax(UInt32) : typemax(UInt32)<<(31-range)
  if degree>=32
   pos=(Atomix.@atomic :monotonic sz[1]+=Int32(1))
   @inbounds wl[pos]=Int32(v)
  end
 end
 if v<e÷32+1
  @inbounds pc2[v+1]=typemax(UInt32)
 end
 return
end
@inline function ecl_large!(w,n,idx,list2,pc,pc2,col,wl,sz)
 @inbounds stop=Int(sz[1])
 if w<stop
  @inbounds v=Int(wl[w+1]);beg=Int(idx[v+1])
  while true
   data=(Atomix.@atomic :monotonic col[v+1]);range=Int(data>>16)
   range==0&&break
   mincol=Int(data&UInt32(0xffff));maxcol=mincol+range;offs=beg÷32
   @inbounds pcol=pc[v+1]
   shortcut=true;done=true
   for k in beg:beg+maxcol-1
    @inbounds nei=Int(list2[k+1])
    neidata=(Atomix.@atomic :monotonic col[nei+1]);neirange=Int(neidata>>16)
    if neirange==0
     neicol=Int(neidata)
     if neicol<32
      pcol&=~(ECL_MSB>>neicol)
     elseif mincol<=neicol<maxcol
      word=offs+neicol÷32+1;mask=ECL_MSB>>(neicol%32)
      old=(Atomix.@atomic :monotonic pc2[word])
      if (old&mask)!=0;Atomix.@atomic :monotonic pc2[word]&=~mask;end
     end
    else
     done=false;neimin=Int(neidata&UInt32(0xffff));neimax=neimin+neirange
     if neimin<=mincol<=neimax;shortcut=false;end
    end
   end
   val=pcol;mc=0
   if pcol==0
    mc=max(1,mincol÷32)
    while true
     val=(Atomix.@atomic :monotonic pc2[offs+mc+1]);val!=0&&break;mc+=1
    end
   end
   newmin=mc*32+leading_zeros(val)
   if mincol!=newmin;shortcut=false;end
   if shortcut||done
    pcol=newmin<32 ? ECL_MSB>>newmin : UInt32(0);newdata=UInt32(newmin)
   else
    newdata=(UInt32(maxcol-newmin)<<16)|UInt32(newmin)
   end
   @inbounds pc[v+1]=pcol
   Atomix.@atomic :monotonic col[v+1]=newdata
  end
 end
 return
end
@inline function ecl_small!(v,n,idx,list,pc,col)
 if v<n
  while true
   pcol=(Atomix.@atomic :monotonic pc[v+1]);count_ones(pcol)<=1&&break
   @inbounds beg=Int(idx[v+1]);active=col[v+1]
   allnei=UInt32(0);keep=active
   while true
    old=active;active&=active-UInt32(1);curr=old⊻active;k=beg+leading_zeros(curr)
    @inbounds nei=Int(list[k+1]);neipcol=(Atomix.@atomic :monotonic pc[nei+1])
    allnei|=neipcol
    if (pcol&neipcol)==0
     pcol&=pcol-UInt32(1);keep⊻=curr
    elseif count_ones(neipcol)==1
     pcol⊻=neipcol;keep⊻=curr
    end
    active==0&&break
   end
   if keep!=0
    best=ECL_MSB>>leading_zeros(pcol)
    if (best&~allnei)!=0;pcol=best;keep=UInt32(0);end
   end
   newcolor=keep==0 ? UInt32(leading_zeros(pcol)) : keep
   @inbounds col[v+1]=newcolor
   Atomix.@atomic :monotonic pc[v+1]=pcol
   keep==0&&break
  end
 end
 return
end
