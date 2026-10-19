# Source graphB+ BSD/Texas State 2021; translated complete active pipeline.
# Volatile LLVM helper pattern reused from BH. Explicit CUDA warp/atomic backend extensions.
@generated function gvload(p::Core.LLVMPtr{Int32,A})where A
 ir="define i32 @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace($A)\n %v = load volatile i32, ptr addrspace($A) %p, align 4\n ret i32 %v }"
 :(Base.llvmcall(($ir,"entry"),Int32,Tuple{UInt},UInt(p)))
end
@generated function gvstore(p::Core.LLVMPtr{Int32,A},x::Int32)where A
 ir="define void @entry(i64 %x, i32 %v) { %p = inttoptr i64 %x to ptr addrspace($A)\n store volatile i32 %v, ptr addrspace($A) %p, align 4\n ret void }"
 :(Base.llvmcall(($ir,"entry"),Cvoid,Tuple{UInt,Int32},UInt(p),x))
end
@inline gg(a,i)=@inbounds a[i+Int32(1)]
@inline gs(a,i,x)=(@inbounds a[i+Int32(1)]=x)
@inline vg(a,i)=gvload(pointer(a,i+Int32(1)))
@inline vs(a,i,x)=gvstore(pointer(a,i+Int32(1)),Int32(x))
@inline ga(a,i,x)=CUDA.atomic_add!(pointer(a,i+Int32(1)),x)
@inline cas(a,i,x,y)=CUDA.atomic_cas!(pointer(a,i+Int32(1)),x,y)
@inline vote(x)=CUDA.vote_ballot_sync(UInt32(0xffffffff),x)
@inline anylane(x)=CUDA.vote_any_sync(UInt32(0xffffffff),x)
@inline shfl(x,l)=CUDA.shfl_sync(UInt32(0xffffffff),x,l+Int32(1))
@inline ws()=CUDA.sync_warp(UInt32(0xffffffff))
@inline lowerball(x,l)=count_ones(x & ~(typemax(UInt32)<<l))%Int32
@inline function gh(v::UInt32)
 v=((v>>16)⊻v)*UInt32(0x45d9f3b);v=((v>>16)⊻v)*UInt32(0x45d9f3b);(v>>16)⊻v
end
@inline function rep(v,label)
 curr=gg(label,v)
 if curr!=v
  prev=v;next=gg(label,curr)
  while curr>next
   gs(label,prev,next);prev=curr;curr=next;next=gg(label,curr)
  end
 end
 curr
end
function stage1!(d,n,e,root,start,stop,level,seed,tid,inc)
 j=tid
 while j<e;gs(d.list,j,gg(d.list,j)<<1);gs(d.info,2j+Int32(1),Int32(gg(d.wei,j)==-1));gs(d.inTree,j,Int32(0));gs(d.neg,j,Int32(0));j+=inc;end
 j=tid;while j<n;gs(d.inCC,j,Int32(0));j+=inc;end;nothing
end
function stage2!(d,n,e,root,start,stop,level,seed,tid,inc)
 j=tid;while j<e;gs(d.list,j,gg(d.list,j)&Int32(-2));j+=inc;end
 j=tid;while j<n;gs(d.parent,j,j==root ? typemax(Int32)&Int32(-4) : Int32(-1));gs(d.label,j,Int32(1));j+=inc;end
 if tid==0;gs(d.queue,Int32(0),root);gs(d.tail,Int32(0),Int32(1));end;nothing
end
function stage3!(d,n,e,root,start,stop,level,seed,tid,inc)
 lane=tid%Int32(32);i=start+tid÷Int32(32);wi=inc÷Int32(32);seed2=seed*seed+seed;bit=(level&Int32(1))|Int32(2)
 while i<stop
  node=gg(d.queue,i);me=(node<<2)|bit;lane==0&&CUDA.atomic_and!(pointer(d.parent,node+Int32(1)),Int32(-4));j=gg(d.idx,node+Int32(1))-Int32(1)-lane
  while j>=gg(d.idx,node)
   neighbor=gg(d.list,j)>>1;seed3=neighbor⊻seed2;hm=reinterpret(Int32,gh(reinterpret(UInt32,me⊻seed3)));val=Int32(0)
   while true
    val=gg(d.parent,neighbor);hv=reinterpret(Int32,gh(reinterpret(UInt32,val⊻seed3)))
    change=val<0||((val&Int32(3))==bit&&(hv<hm||(hv==hm&&val<me)))
    (!change||cas(d.parent,neighbor,val,me)==val)&&break
   end
   if val<0;p=ga(d.tail,Int32(0),Int32(1));gs(d.queue,p,neighbor);end;j-=Int32(32)
  end
  ws();i+=wi
 end;nothing
end
function stage4!(d,n,e,root,start,stop,level,seed,tid,inc)
 i=start+tid;while i<stop;node=gg(d.queue,i);ga(d.label,gg(d.parent,node)>>2,gg(d.label,node));i+=inc;end;nothing
end
@inline function swapedge!(d,a,b)
 x=vg(d.list,a);y=vg(d.list,b);vs(d.list,a,y);vs(d.list,b,x)
 for f in (d.inTree,d.neg);x=vg(f,a);y=vg(f,b);vs(f,a,y);vs(f,b,x);end
 for k in Int32(0):Int32(1);x=gg(d.info,2a+k);y=gg(d.info,2b+k);gs(d.info,2a+k,y);gs(d.info,2b+k,x);end;nothing
end
function stage5!(d,n,e,root,start,stop,level,seed,tid,inc)
 lane=tid%Int32(32);i=start+tid÷Int32(32);wi=inc÷Int32(32)
 while i<stop
  node=gg(d.queue,i);par=gg(d.parent,node)>>2;nl=gg(d.label,node);beg=gg(d.idx,node);last=gg(d.idx,node+Int32(1));lbl=(nl>>1)+Int32(1);j=beg+lane
  while anylane(j<last)
   val=Int32(0);cond=false
   if j<last;nei=vg(d.list,j)>>1;cond=nei!=par&&(gg(d.parent,nei)>>2)==node;cond&&(val=gg(d.label,nei));end
   cc=val;delta=Int32(1)
   while delta<Int32(32);tmp=CUDA.shfl_up_sync(UInt32(0xffffffff),val,delta);lane>=delta&&(val+=tmp);delta*=Int32(2);end
   lbl+=val
   if cond;lv=(lbl-cc)<<1;gs(d.label,nei,lv);gs(d.info,2j,lv);gs(d.info,2j+Int32(1),(gg(d.info,2j+Int32(1))&Int32(1))|((lbl-Int32(1))<<1));vs(d.list,j,vg(d.list,j)|Int32(1));end
   lbl=shfl(lbl,Int32(31));j+=Int32(32)
  end
  len=last-beg
  if len>0
   if len<=Int32(32)
    src=beg+lane;b=Int32(0);en=Int32(0);iv=Int32(0);neg=Int32(0);nv=Int32(0);state=Int32(0)
    if lane<len
     b=gg(d.info,2src);en=gg(d.info,2src+Int32(1));iv=vg(d.inTree,src);neg=vg(d.neg,src);nv=vg(d.list,src);nei=nv>>1;state=nei!=par&&(gg(d.parent,nei)>>2)==node ? Int32(2) : Int32(3)
    end
    ball=vote(state==2);balr=vote(state==3);pfsl=lowerball(ball,lane);pfsr=lowerball(balr,lane);pos=beg+(state==3 ? len-Int32(1)-pfsr : pfsl)
    if state!=0;gs(d.info,2pos,b);gs(d.info,2pos+Int32(1),en);vs(d.inTree,pos,iv);vs(d.neg,pos,neg);vs(d.list,pos,nv);end
   else
    lp=beg;rp=last-Int32(1);state=Int32(1);read=beg+min(Int32(32),len);src=beg+lane;b=gg(d.info,2src);en=gg(d.info,2src+Int32(1));nv=vg(d.list,src);iv=vg(d.inTree,src);neg=vg(d.neg,src)
    while true
     if state==1;nei=nv>>1;state=nei!=par&&(gg(d.parent,nei)>>2)==node ? Int32(2) : Int32(3);end
     ball=vote(state==2);pfsl=lowerball(ball,lane)
     if state==2
      pos=lp+pfsl;ob=Int32(0);oe=Int32(0);oi=Int32(0);og=Int32(0);on=Int32(0)
      if pos>=read;ob=gg(d.info,2pos);oe=gg(d.info,2pos+Int32(1));oi=vg(d.inTree,pos);og=vg(d.neg,pos);on=vg(d.list,pos);end
      gs(d.info,2pos,b);gs(d.info,2pos+Int32(1),en);vs(d.inTree,pos,iv);vs(d.neg,pos,neg);vs(d.list,pos,nv);b=ob;en=oe;iv=oi;neg=og;nv=on;state=pos<read ? Int32(0) : Int32(1)
     end
     lp+=count_ones(ball)%Int32;read=max(read,lp);balr=vote(state==3);pfsr=lowerball(balr,lane)
     if state==3
      pos=rp-pfsr;ob=Int32(0);oe=Int32(0);oi=Int32(0);og=Int32(0);on=Int32(0)
      if pos>=read;ob=gg(d.info,2pos);oe=gg(d.info,2pos+Int32(1));oi=vg(d.inTree,pos);og=vg(d.neg,pos);on=vg(d.list,pos);end
      gs(d.info,2pos,b);gs(d.info,2pos+Int32(1),en);vs(d.inTree,pos,iv);vs(d.neg,pos,neg);vs(d.list,pos,nv);b=ob;en=oe;iv=oi;neg=og;nv=on;state=pos<read ? Int32(0) : Int32(1)
     end
     rp-=count_ones(balr)%Int32
     if read<=rp
      bal=vote(state==0);pfs=lowerball(bal,lane)
      if state==0
       pos=read+pfs
       if pos<=rp;b=gg(d.info,2pos);en=gg(d.info,2pos+Int32(1));iv=vg(d.inTree,pos);neg=vg(d.neg,pos);nv=vg(d.list,pos);state=Int32(1);end
      end
      read+=count_ones(bal)%Int32
     end
     !anylane(state==1)&&break
    end
   end
  end
  pe=Int32(-1);j=beg+lane
  while anylane(j<last)
   if j<last&&(vg(d.list,j)>>1)==par;pe=j;end
   anylane(pe>=0)&&break;j+=Int32(32)
  end
  pos=Int32(-1);j=beg+lane
  while anylane(j<last)
   if j<last;nei=vg(d.list,j)>>1;if(gg(d.parent,nei)>>2)!=node;pos=j;end;end
   anylane(pos>=0)&&break;j+=Int32(32)
  end
  bal=vote(pos>=0);lid=Int32(trailing_zeros(bal));pos=shfl(pos,lid)
  if pe>=0
   gs(d.info,2pe,nl|Int32(1));gs(d.info,2pe+Int32(1),(gg(d.info,2pe+Int32(1))&Int32(1))|((lbl-Int32(1))<<1));vs(d.list,pe,vg(d.list,pe)|Int32(1))
   if pe!=beg
    if pe!=pos;swapedge!(d,pos,pe);pe=pos;end
    pe!=beg&&swapedge!(d,beg,pe)
   end
  end
  ws();i+=wi
 end;nothing
end
function stage6!(d,n,e,root,start,stop,level,seed,tid,inc)
 j=tid;while j<e;vs(d.inTree,j,vg(d.inTree,j)+(gg(d.list,j)&Int32(1)));j+=inc;end;nothing
end
function stage7!(d,n,e,root,start,stop,level,seed,tid,inc)
 j=tid;while j<e;gs(d.minus,j,UInt8(1));j+=inc;end
 u=tid;while u<n;j=gg(d.idx,u);while j<gg(d.idx,u+Int32(1))&&(gg(d.list,j)&Int32(1))!=0;gs(d.minus,j,UInt8(gg(d.info,2j+Int32(1))&Int32(1)));j+=Int32(1);end;u+=inc;end;nothing
end
function stage8!(d,n,e,root,start,stop,level,seed,tid,inc)
 lane=tid%Int32(32);u=tid÷Int32(32);wi=inc÷Int32(32)
 while u<n
  target=gg(d.label,u);target1=target|Int32(1);j=gg(d.idx,u+Int32(1))-Int32(1)-lane
  while j>=gg(d.idx,u)&&(gg(d.list,j)&Int32(1))==0
   curr=gg(d.list,j)>>1
   if curr>u
    total=Int32(0)
    while gg(d.label,curr)!=target
     k=gg(d.idx,curr)
     while (gg(d.info,2k)&Int32(1))==Int32(gg(d.info,2k)<=target1&&target<=gg(d.info,2k+Int32(1)));k+=Int32(1);end
     total+=gg(d.info,2k+Int32(1))&Int32(1);curr=gg(d.list,k)>>1
    end
    gs(d.minus,j,UInt8(total&Int32(1)))
   end
   j-=Int32(32)
  end
  ws();u+=wi
 end;nothing
end
function stage9!(d,n,e,root,start,stop,level,seed,tid,inc)
 v=tid;while v<n;gs(d.label,v,v);gs(d.queue,v,Int32(0));v+=inc;end;nothing
end
function stage10!(d,n,e,root,start,stop,level,seed,tid,inc)
 lane=tid%Int32(32);v=tid÷Int32(32);wi=inc÷Int32(32)
 while v<n
  beg=gg(d.idx,v);last=gg(d.idx,v+Int32(1));vstat=rep(v,d.label);j=beg+lane
  while j<last
   nli=gg(d.list,j)>>1
   if gg(d.minus,j)!=0;gs(d.neg,j,gg(d.neg,j)+Int32(1))
   else
    ostat=rep(nli,d.label)
    while true
     repeat=false
     if vstat!=ostat
      if vstat<ostat;ret=cas(d.label,ostat,ostat,vstat);ret!=ostat&&(ostat=ret;repeat=true)
      else;ret=cas(d.label,vstat,vstat,ostat);ret!=vstat&&(vstat=ret;repeat=true);end
     end
     !repeat&&break
    end
   end
   j+=Int32(32)
  end
  v+=wi
 end;nothing
end
function stage11!(d,n,e,root,start,stop,level,seed,tid,inc)
 v=tid;while v<n;x=gg(d.label,v);old=x;next=gg(d.label,x);while x>next;x=next;next=gg(d.label,x);end;old!=x&&gs(d.label,v,x);v+=inc;end;nothing
end
function stage12!(d,n,e,root,start,stop,level,seed,tid,inc)
 if tid==0;gs(d.hi,Int32(0),UInt64(0));gs(d.wsize,Int32(0),Int32(0));end
 v=tid;while v<n;ga(d.queue,gg(d.label,v),Int32(1));v+=inc;end;nothing
end
function stage13!(d,n,e,root,start,stop,level,seed,tid,inc)
 v=tid;while v<n;x=(UInt64(gg(d.queue,v))<<32)|UInt64(v);gg(d.hi,Int32(0))<x&&CUDA.atomic_max!(pointer(d.hi),x);v+=inc;end;nothing
end
function stage14!(d,n,e,root,start,stop,level,seed,tid,inc)
 lane=tid%Int32(32);v=tid÷Int32(32);wi=inc÷Int32(32);hi=Int32(gg(d.hi,Int32(0))&UInt64(0xffffffff))
 while v<n
  lv=gg(d.label,v);lv==v&&gs(d.queue,lv,lv==hi ? Int32(0) : typemax(Int32)-Int32(1));j=gg(d.idx,v)+lane
  while j<gg(d.idx,v+Int32(1));ln=gg(d.label,gg(d.list,j)>>1);if lv<ln;k=ga(d.wsize,Int32(0),Int32(1));gs(d.ws1,k,lv);gs(d.ws2,k,ln);end;j+=Int32(32);end;v+=wi
 end;nothing
end
function stage15!(d,n,e,root,start,stop,level,seed,tid,inc)
 i=tid;while i<gg(d.wsize,Int32(0));v=gg(d.ws1,i);u=gg(d.ws2,i);dv=gg(d.queue,v);du=gg(d.queue,u);if dv+Int32(1)<du;gs(d.queue,u,dv+Int32(1));gs(d.changed,Int32(0),UInt8(1));elseif du+Int32(1)<dv;gs(d.queue,v,du+Int32(1));gs(d.changed,Int32(0),UInt8(1));end;i+=inc;end;nothing
end
function stage16!(d,n,e,root,start,stop,level,seed,tid,inc)
 v=tid;while v<n;gs(d.inCC,v,gg(d.inCC,v)+((gg(d.queue,gg(d.label,v))%Int32(2))⊻Int32(1)));v+=inc;end;nothing
end
