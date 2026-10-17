const WM=typemax(UInt32)
const KM=UInt32(0xaaaaaaa8)
const PM=UInt32(0x55555554)
const HI=UInt32(0x80000000)
const LO=UInt32(0x7fffffff)
@inline lane()=(Int32(CUDA.threadIdx().x)-Int32(1))&Int32(31)
@inline sh(x,l)=CUDA.shfl_sync(WM,x,Int32(l)+Int32(1))
@inline bal(x)=CUDA.vote_ballot_sync(WM,x)
@inline ffs(x::UInt32)=Int32(trailing_zeros(x))+Int32(1)
@inline getword(pool,node,l)=@inbounds pool[Int32(node)*Int32(32)+Int32(l)+Int32(1)]
@inline function putword!(pool,node,l,v)
 @inbounds pool[Int32(node)*Int32(32)+Int32(l)+Int32(1)]=v
 return
end
@inline rd(pool,node)=getword(pool,node,Int32(31)-lane())
@inline wr!(pool,node,v)=putword!(pool,node,Int32(31)-lane(),v)
@inline function alloc!(count)
 CUDA.atomic_add!(pointer(count,1),UInt32(1))
end
@inline function trylock!(pool,node)
 locked=UInt32(1)
 if lane()==Int32(1);locked=CUDA.atomic_or!(pointer(pool,Int32(node)*Int32(32)+Int32(2)),HI)&HI;end
 locked=sh(locked,Int32(1))
 if locked!=0;return true;end
 CUDA.threadfence();return false
end
@inline function lock!(pool,node)
 locked=true
 while locked
  x=UInt32(0)
  if lane()==Int32(1);x=CUDA.atomic_or!(pointer(pool,Int32(node)*Int32(32)+Int32(2)),HI)&HI;end
  locked=sh(x,Int32(1))!=0
 end
 CUDA.threadfence();return
end
@inline function unlock!(pool,node)
 CUDA.threadfence()
 if lane()==Int32(1);CUDA.atomic_and!(pointer(pool,Int32(node)*Int32(32)+Int32(2)),LO);end
 return
end
@inline function insertnode!(pool,intermediate,node,key,value,data)
 if intermediate;data=rd(pool,node);end
 data&=LO;key&=LO
 location=bal((key>data)&&(data!=0))&KM
 dest=location==0 ? Int32(33) : ffs(location)-Int32(1)
 # Native unsigned shifts use modulo32; negative shifts do not occur for selected pivots.
 tomove=((UInt32(1)<<(dest-Int32(3)))-UInt32(1))&UInt32(0xfffffffc)
 mask=UInt32(1)<<lane();moved=CUDA.shfl_down_sync(WM,data,Int32(2))
 newdata=(mask&tomove)!=0 ? moved : data
 if ((UInt32(1)<<(dest-Int32(2)))&mask)!=0;newdata=key;end
 if ((UInt32(1)<<(dest-Int32(3)))&mask)!=0;newdata=value;end
 valid=reinterpret(UInt32,reinterpret(Int32,bal(data!=0))>>Int32(2))
 if intermediate&&((valid&mask&KM)!=0);newdata|=HI;end
 if lane()==Int32(30);newdata|=HI;end
 wr!(pool,node,newdata);return
end
@inline function splitnode!(pool,count,parent,key,node,data)
 right=CUDA.shfl_up_sync(UInt32(0x0000ffff),data,Int32(16))
 if lane()<Int32(18);right=UInt32(0);end
 if lane()<Int32(2);right=data;end
 rightmin=sh(right,Int32(31))&LO
 lock!(pool,parent);pdata=rd(pool,parent)
 if sh(pdata,Int32(3))!=0
  unlock!(pool,node);unlock!(pool,parent);return true,node,data
 end
 mine=bal((pdata&LO)==node)&PM
 if mine==0
  unlock!(pool,node);unlock!(pool,parent);return true,node,data
 end
 ri=UInt32(0)
 if lane()==0;ri=alloc!(count);end
 ri=sh(ri,Int32(0));lock!(pool,ri)
 insertnode!(pool,true,parent,rightmin,ri,pdata);unlock!(pool,parent)
 if lane()==Int32(30);right|=HI;end
 wr!(pool,ri,right)
 if lane()<Int32(16);data=UInt32(0);end
 if lane()==Int32(1);data=rightmin;end
 if lane()==Int32(0);data=ri;end
 if lane()==Int32(30);data|=HI;end
 wr!(pool,node,data)
 if key>=rightmin
  unlock!(pool,node);data=right;node=ri
 else
  unlock!(pool,ri)
 end
 return false,node,data&LO
end
@inline function splitroot!(pool,count,key,node,data)
 li=UInt32(0);ri=UInt32(0)
 if lane()==0;li=alloc!(count);ri=alloc!(count);end
 li=sh(li,Int32(0));ri=sh(ri,Int32(0));lock!(pool,li);lock!(pool,ri)
 rootdata=sh(data,Int32(15))|HI;rev=Int32(31)-lane()
 if rev==0;rootdata=data|HI
 elseif rev==1;rootdata=li|HI
 elseif rev==3;rootdata=ri
 elseif rev!=2;rootdata=UInt32(0)
 end
 wr!(pool,node,rootdata);unlock!(pool,node)
 rightmin=sh(rootdata,Int32(29))&LO;left=data
 if lane()<Int32(16);left=UInt32(0);end
 if lane()==Int32(1);left=rightmin;end
 if lane()==Int32(0);left=ri;end
 if lane()<Int32(16)
  if lane()==Int32(14);data|=HI;end
  putword!(pool,ri,Int32(15)-lane(),data)
 end
 if lane()==Int32(30);left|=HI;end
 wr!(pool,li,left)
 if rev==0||rev==2;rootdata&=LO;end
 go=true
 if rev==2&&key<rootdata;go=false;end
 go=sh(go,Int32(29))
 if go
  unlock!(pool,li);data=CUDA.shfl_up_sync(UInt32(0x0000ffff),data,Int32(16))
  if lane()<Int32(16);data=UInt32(0);end
  node=ri
 else
  unlock!(pool,ri);data=left;node=li
 end
 return node,data&LO
end
@inline function insertunit!(pool,count,root,key,value,active)
 last=UInt32(0);ra=@inbounds root[1];parent=ra;next=ra
 while true
  work=bal(active)
  if work==0;break;end
  flags=Int32(0);src=sh(key,ffs(work)-Int32(1))
  if last!=work;next=ra;parent=ra;end
  data=rd(pool,next);link=sh(data,Int32(1))&LO
  while link!=0&&src>=link
   next=sh(data,Int32(0))&LO;data=rd(pool,next);link=sh(data,Int32(1))&LO;flags|=Int32(2)
  end
  first=sh(data,Int32(31));flags=(first&HI)==0 ? flags|Int32(4) : flags&Int32(3)
  if (flags&4)!=0
   if trylock!(pool,next);next=parent;continue;end
   data=rd(pool,next);first=sh(data,Int32(31));flags=(first&HI)==0 ? flags|Int32(4) : flags&Int32(3);link=sh(data,Int32(1))&LO
   if parent==ra&&link!=0&&src>=link
    unlock!(pool,next);next=ra;parent=ra;continue
   end
   if (flags&4)==0;unlock!(pool,next);end
   while link!=0&&src>=link
    if (flags&4)!=0;unlock!(pool,next);end
    next=sh(data,Int32(0))&LO
    if (flags&4)!=0;lock!(pool,next);end
    data=rd(pool,next);first=sh(data,Int32(31));flags=(first&HI)==0 ? flags|Int32(4) : flags&Int32(3)
    if (flags&4)==0;unlock!(pool,next);end
    link=sh(data,Int32(1))&LO;flags|=Int32(2)
   end
  end
  flags=sh(data,Int32(3))!=0 ? flags|Int32(8) : flags&Int32(7)
  if (flags&2)!=0&&(flags&8)!=0
   if (flags&4)!=0;unlock!(pool,next);next=ra;parent=ra;continue
   else;flags&=Int32(7);end
  end
  if (flags&8)!=0&&(flags&4)==0
   if trylock!(pool,next);next=parent;continue
   else
    data=rd(pool,next);flags=sh(data,Int32(3))!=0 ? flags|Int32(8) : flags&Int32(7)
    if (flags&8)!=0
     link=sh(data,Int32(1))&LO
     if link!=0&&src>=link;unlock!(pool,next);next=ra;parent=ra;continue;end
    else
     unlock!(pool,next);link=sh(data,Int32(1))&LO
     while link!=0&&src>=link
      next=sh(data,Int32(0))&LO;data=rd(pool,next);link=sh(data,Int32(1))&LO;flags|=Int32(2)
     end
    end
   end
  end
  if (flags&8)!=0&&next!=ra&&parent==next
   unlock!(pool,next);next=ra;parent=ra;continue
  end
  if (flags&8)!=0&&next!=ra
   retry,next,data=splitnode!(pool,count,parent,src,next,data)
   if retry;next=ra;parent=ra;continue;end
   if (flags&4)==0;unlock!(pool,next);end
  elseif (flags&8)!=0
   next,data=splitroot!(pool,count,src,next,data);flags|=Int32(1)
   if (flags&4)==0;unlock!(pool,next);end
  end
  parent=(flags&1)!=0 ? ra : next
  if (flags&4)!=0
   sl=ffs(work)-Int32(1);sv=sh(value,sl);exists=(bal(src==data)&KM)!=0
   if !exists;insertnode!(pool,false,next,src,sv,data);end
   unlock!(pool,next)
   if sl==lane();active=false;end
  else
   data=data==0 ? WM : data;unitkey=data&LO;pivot=bal(src>=unitkey)&KM
   next=pivot==0 ? sh(unitkey,Int32(30)) : sh(unitkey,ffs(pivot)-Int32(2))
  end
  last=work
 end
 return
end
@inline function searchunit(pool,root,key,active)
 ra=@inbounds root[1];last=UInt32(0);next=ra;result=UInt32(0)
 while true
  work=bal(active)
  if work==0;break;end
  sl=ffs(work)-Int32(1);src=sh(key,sl)
  if last!=work;next=ra;end
  data=rd(pool,next);leaf=sh((data&HI)==0,Int32(31));data=data==0 ? WM : data;unitkey=data&LO;pivot=bal(src>=unitkey)&KM
  dest=pivot==0 ? Int32(-1) : ffs(pivot)-Int32(1)
  if dest<0
   if lane()==sl;result=UInt32(0);active=false;end
  else
   next=sh(data,dest-Int32(1));found=sh(leaf&&data==src,dest)
   if found&&lane()==sl;result=next;active=false;end
   if leaf&&!found&&lane()==sl;result=UInt32(0);active=false;end
  end
  last=work
 end
 return result
end
