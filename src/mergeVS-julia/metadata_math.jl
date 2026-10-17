# Microsoft MIT mergeVS-derived source contract. One row/head/batch per work item.
@inline function save_metadata_blocks!(out,base,rs,re,bs,count,kv)
 if rs>=kv;return count;end
 re=min(re,kv)
 for k in rs:bs:re-1
  @inbounds out[base+count+1]=Int32(k)
  count+=1
 end
 count
end
@inline function metadata_row!(row,h,b,q,kv,vertical,slash,vc,sc,bc,bo,cc,ci,H,R,M,BN,V,S,causal)
 @inbounds qlen=Int64(q[b+1]);kvlen=Int64(kv[b+1]);start=row*M;stop=start+M
 if start<qlen
  inputbase=(b*H+h);vbase=inputbase*V;sbase=inputbase*S;outrow=inputbase*R+row
  @inbounds nv=Int64(vc[h+1]);ns=Int64(sc[h+1]);vi=Int64(vertical[vbase+1]);si=Int64(slash[sbase+1])
  v=Int64(1);s=Int64(1);nb=Int64(0);nc=Int64(0);delta=causal ? kvlen-qlen : kvlen
  while si>=stop+delta&&s<ns
   @inbounds si=Int64(slash[sbase+s+1]);s+=1
  end
  has=si<=stop+delta;si=max(delta+stop-si,M);rs=si-M;re=si
  if !has
   rs=causal ? delta+stop : kvlen;re=rs+BN
  end
  finished=false
  while true
   if vi<re
    if vi<rs
     @inbounds ci[outrow*V+nc+1]=Int32(vi);nc+=1
    end
    if v<nv
     @inbounds vi=Int64(vertical[vbase+v+1]);v+=1
    else
     vi=stop+BN+delta
    end
   else
    @inbounds can_next=s<ns&&(causal||slash[sbase+s+1]>=start)
    if can_next
     @inbounds si=max(delta+stop-Int64(slash[sbase+s+1]),M);s+=1
    else
     if v==nv||(vi>rs&&causal)
      if v==nv&&!causal&&vi<kvlen
       @inbounds ci[outrow*V+nc+1]=Int32(vi);nc+=1
      end
      nb=save_metadata_blocks!(bo,outrow*S,rs,re,BN,nb,kvlen)
      break
     else
      if causal
       rs=delta+stop;re=rs+BN
      else
       nb=save_metadata_blocks!(bo,outrow*S,rs,re,BN,nb,kvlen);rs=kvlen;re=kvlen+BN
      end
      finished=true
     end
    end
    if !finished
     if si>re+M
      nb=save_metadata_blocks!(bo,outrow*S,rs,re,BN,nb,kvlen);rs=si-M;re=si
     elseif si>re
      re+=M
     end
    end
   end
  end
  @inbounds bc[outrow+1]=Int32(nb);cc[outrow+1]=Int32(nc)
 end
 return
end
