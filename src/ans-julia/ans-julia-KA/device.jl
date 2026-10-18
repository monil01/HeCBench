# Original MULTIANS arithmetic/control transcribed under LGPL-3.0.
# UInt32 sliding-window/state values; no floating-point decoding arithmetic.
struct SyncPoint
 x::UInt32
 y::UInt32
 z::UInt32
 w::UInt32
end
@inline function decode_subsequence(subsize,cur,processed,startbit,inpos,IN,window,next,state,lastunit,lastbit,nums,outpos,OUT,nextout,TABLE,lastat,laststate,overflow::Bool,writeout::Bool)
 unit=UInt32(0);at=startbit;ns=UInt32(0);copy_next=next;mask=typemax(UInt32)
 if cur==0 || processed==0
  copy_next<<=UInt32(32)-at;next>>=at;window>>=at;window|=copy_next
 end
 if overflow && cur>0 && processed==0
  hit=@inbounds TABLE[state-UInt32(1024)+UInt32(1)];following=hit&UInt32(0xffff);taken=hit>>UInt32(24)
  state=(following<<taken)|(~(mask<<taken)&window)
  while state<1024
   shifted=window>>taken;taken+=UInt32(1);state=(state<<UInt32(1))|(~(mask<<UInt32(1))&shifted)
  end
  if at==0
   ns+=UInt32(1)
   if writeout && outpos<nextout;@inbounds OUT[outpos+UInt32(1)]=UInt8((hit>>UInt32(16))&UInt32(255));outpos+=UInt32(1);end
  end
  copy_next=taken>0 ? next<<(UInt32(32)-taken) : UInt32(0)
  next>>=taken;window>>=taken;at+=taken;window|=copy_next
  if at>32
   inpos+=UInt32(1);@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];copy_next=next;at-=UInt32(32);window>>=at;next>>=at;copy_next<<=UInt32(32)-at;window|=copy_next
  end
  if at==32
   inpos+=UInt32(1);@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];copy_next=next;at=UInt32(0)
  end
 end
 while unit<subsize
  while at<32
   laststate=state;hit=@inbounds TABLE[state-UInt32(1024)+UInt32(1)];following=hit&UInt32(0xffff);taken=hit>>UInt32(24)
   state=(following<<taken)|(~(mask<<taken)&window)
   while state<1024
    shifted=window>>taken;taken+=UInt32(1);state=(state<<UInt32(1))|(~(mask<<UInt32(1))&shifted)
   end
   ns+=UInt32(1)
   if writeout && outpos<nextout;@inbounds OUT[outpos+UInt32(1)]=UInt8((hit>>UInt32(16))&UInt32(255));outpos+=UInt32(1);end
   copy_next=taken>0 ? next<<(UInt32(32)-taken) : UInt32(0)
   next>>=taken;window>>=taken;lastbit=at;at+=taken;window|=copy_next;lastunit=unit
  end
  unit+=UInt32(1);inpos+=UInt32(1);@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];copy_next=next
  if at==32;at=UInt32(0)
  else;at-=UInt32(32);window>>=at;next>>=at;copy_next<<=UInt32(32)-at;window|=copy_next
  end
 end
 (inpos,window,next,state,lastunit,lastbit,ns,outpos,at,laststate)
end
@inline function ans_body(::Val{OP},IN,TABLE,SYNC,AUX,FLAGS,OUT,n,blocks,first,firstbit,outbytes,gid,group,width)where OP
 if OP==0
  if gid*UInt32(4)<n
   cur=gid*UInt32(4);inpos=gid*UInt32(16);state=first;@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)]
   lastunit=UInt32(0);lastbit=UInt32(0);ns=UInt32(0);outpos=UInt32(0);nextout=UInt32(0);lastat=gid==0 ? UInt32(32)-firstbit : UInt32(0);laststate=UInt32(0);processed=UInt32(0);synced=false;lastsub=min(width*(group+UInt32(1))*UInt32(4),n)
   while processed<width*UInt32(4)
    # Four writes are deferred until allfour source subcalls/sync-tests complete.
    s0=SyncPoint(0,0,0,0);s1=s0;s2=s0;s3=s0;w0=false;w1=false;w2=false;w3=false
    for lane=UInt32(0):UInt32(3)
     if !synced && cur<lastsub
      inpos,window,next,state,lastunit,lastbit,ns,outpos,lastat,laststate=decode_subsequence(UInt32(4),cur+lane,processed+lane,lastat,inpos,IN,window,next,state,lastunit,lastbit,ns,outpos,OUT,nextout,TABLE,lastat,laststate,false,false)
      if processed>=4
       point=@inbounds SYNC[cur+lane+UInt32(1)]
       if point.x==lastunit && point.y==lastbit && point.w==laststate;synced=true;end
      end
      point=SyncPoint(lastunit,lastbit,ns,laststate)
      if lane==0;s0=point;w0=true;elseif lane==1;s1=point;w1=true;elseif lane==2;s2=point;w2=true;else;s3=point;w3=true;end
     end
    end
    if w0;@inbounds SYNC[cur+UInt32(1)]=s0;end
    if w1;@inbounds SYNC[cur+UInt32(2)]=s1;end
    if w2;@inbounds SYNC[cur+UInt32(3)]=s2;end
    if w3;@inbounds SYNC[cur+UInt32(4)]=s3;end
    cur+=UInt32(4);processed+=UInt32(4);KernelAbstractions.__synchronize()
   end
  end
 elseif OP==1
  if group<blocks-UInt32(1)
   cur=(group+UInt32(1))*width;point=@inbounds SYNC[cur];inpos=(cur-UInt32(1))*UInt32(4);lastunit=inpos+point.x;lastbit=point.y;state=point.w;laststate=UInt32(0);ns=UInt32(0);lastat=point.y;inpos+=point.x;@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];processed=UInt32(0);synced=false;outpos=UInt32(0)
   while processed<width
    if !synced
     inpos,window,next,state,lastunit,lastbit,ns,outpos,lastat,laststate=decode_subsequence(UInt32(4),cur,processed,lastat,inpos,IN,window,next,state,lastunit,lastbit,ns,outpos,OUT,UInt32(0),TABLE,lastat,laststate,true,false)
     point=@inbounds SYNC[cur+UInt32(1)]
     if point.x==lastunit && point.y==lastbit && point.w==laststate
      point=SyncPoint(point.x,point.y,ns,point.w);@inbounds FLAGS[group+UInt32(2)]=UInt8(1);synced=true
     else
      point=SyncPoint(lastunit,lastbit,ns,laststate);@inbounds FLAGS[group+UInt32(2)]=UInt8(0)
     end
     @inbounds SYNC[cur+UInt32(1)]=point
    end
    cur+=UInt32(1);processed+=UInt32(1);KernelAbstractions.__synchronize()
   end
  end
 elseif OP==2
  if gid<n;@inbounds AUX[gid+UInt32(1)]=SYNC[gid+UInt32(1)].z;end
 elseif OP==3
  if gid<n;@inbounds p=SYNC[gid+UInt32(1)];SYNC[gid+UInt32(1)]=SyncPoint(p.x,p.y,AUX[gid+UInt32(1)],p.w);end
 elseif OP==4
  if gid<n
   point=@inbounds SYNC[gid+UInt32(1)];nextpoint=@inbounds SYNC[gid+UInt32(2)];outpos=point.z;nextout=gid==n-UInt32(1) ? outbytes : nextpoint.z;state=first;inpos=gid*UInt32(4)
   if gid>0;point=@inbounds SYNC[gid];inpos=(gid-UInt32(1))*UInt32(4);state=point.w;end
   @inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];start=UInt32(32)-firstbit
   if gid>0;inpos+=point.x;start=point.y;@inbounds window=IN[inpos+UInt32(1)];next=IN[inpos+UInt32(2)];end
   decode_subsequence(UInt32(4),gid,UInt32(0),start,inpos,IN,window,next,state,UInt32(0),UInt32(0),UInt32(0),outpos,OUT,nextout,TABLE,UInt32(0),UInt32(0),true,true)
  end
 end
 nothing
end
