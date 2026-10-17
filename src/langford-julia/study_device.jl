using Atomix
@inline function push_state!(stack,top,k,m,d,num,base)
 @inbounds begin
 stack[base+top+1]=Int8(k);stack[base+top+2]=Int8(m);stack[base+top+3]=Int8(d);stack[base+top+4]=Int8(num)
 end
 return top+4
end
@inline function dfs_lane!(result,availbuf,openbuf,stack,pos,lid,gid,::Val{N}) where N
 ab=(lid-1)*(2N+1);ob=(lid-1)*(4N+2);sb=(lid-1)*24N;pb=(lid-1)*(8*((N+7)÷8));top=0
 @inbounds begin
 availbuf[ab+1]=Int32((Int64(1)<<N)-1);openbuf[ob+1]=0;openbuf[ob+2]=0
 top=push_state!(stack,top,0,-1,0,0,sb)
 while top>0
  num=Int32(stack[sb+top]);top-=1;side=Int32(stack[sb+top]);top-=1;m=Int32(stack[sb+top]);top-=1;k=Int32(stack[sb+top]);top-=1
  oi=ob+2k+3;openbuf[oi]=openbuf[oi-2];openbuf[oi+1]=openbuf[oi-1];avail=availbuf[ab+k+1]
  if m>=0
   pos[pb+m+1]=Int8(k);avail ⊻=Int32(1)<<m;openbuf[oi+side]&=openbuf[oi+side]-Int64(1)
  else
   openbuf[oi+side]|=(Int64(1)<<(2N-1))>>k;num+=1
  end
  k+=1;availbuf[ab+k+1]=avail
  if k==2N
   index=Atomix.@atomic result[1]+=Int64(1)
   # Atomix returns newvalue for @atomic x+=; storagecounter valueonecorrespondsfirsttuple.
   if index<=4096
    for j in 1:N;result[1+(index-1)*N+j]=Int64(pos[pb+j]);end
   end
  else
   klim=N>19 ? 8+N÷3 : N-5
   hash=reinterpret(UInt64,Int64(131071)*(openbuf[oi+1]-openbuf[oi])+Int64(avail))
   if k==klim&&hash%UInt64(16383)!=UInt64(gid-1);continue;end
   offset=k-2N-2
   for dd in 0:1
    opening=openbuf[oi+dd]
    if opening!=0
     mm=offset+trailing_zeros(reinterpret(UInt64,opening))+1
     if 0<=mm<N&&((avail>>mm)&1)!=0&&(mm!=0||k<=N)
      top=push_state!(stack,top,k,mm,dd,num,sb)
     end
    end
   end
   if num<N
    top=push_state!(stack,top,k,-1,1,num,sb);top=push_state!(stack,top,k,-1,0,num,sb)
   end
  end
 end
 end
 return
end
