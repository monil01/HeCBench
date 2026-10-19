# Georgia Tech hybrid_BC BSD source: exact/default Brandes pipeline and bitonic helper.
# Own frontend entrypoints supply 0-based tid/block, shared control and the barrier/atomic API.
@inline hg(a,i)=@inbounds a[Int(i)+1]
@inline hs(a,i,x)=(@inbounds a[Int(i)+1]=x)
@inline hc(sh,i)=hg(sh,Int32(i))
@inline hcs(sh,i,x)=hs(sh,Int32(i),Int32(x))
@inline hsync(::Val{:cuda_jl})=CUDA.sync_threads()
@inline hfetch(::Val{:cuda_jl},a,i,v)=CUDA.atomic_add!(pointer(a,Int(i)+1),v)
@inline hcas(::Val{:cuda_jl},a,i,old,new)=CUDA.atomic_cas!(pointer(a,Int(i)+1),old,new)
function bitonic!(sh,tid,mode)
 idx=tid;k=Int32(2)
 while k<=Int32(512)
  j=k>>1
  while j>0
   while idx<Int32(512)
    ixj=idx⊻j
    if ixj>idx
     if (idx&k)==0&&hc(sh,Int32(16)+idx)>hc(sh,Int32(16)+ixj)
      tmp=hc(sh,Int32(16)+idx);hcs(sh,Int32(16)+idx,hc(sh,Int32(16)+ixj));hcs(sh,Int32(16)+ixj,tmp)
     end
     if (idx&k)!=0&&hc(sh,Int32(16)+idx)<hc(sh,Int32(16)+ixj)
      tmp=hc(sh,Int32(16)+idx);hcs(sh,Int32(16)+idx,hc(sh,Int32(16)+ixj));hcs(sh,Int32(16)+ixj,tmp)
     end
    end
    idx+=Int32(1024)
   end
   hsync(mode);idx=tid;j>>=1
  end
  k<<=1
 end;nothing
end
function hbc_body!(d,p,n,m,sh,tid,block,mode)
 # Pitched byte pointers become exactly equivalent Int32 element offsets within named sizes.
 od=block*p.dist;os=block*p.sigma;ode=block*p.delta;oq=block*p.Q;oq2=block*p.Q2;ost=block*p.S;oep=block*p.ep
 if tid==0
  hcs(sh,0,block);hcs(sh,1,block);hs(d.jia,Int32(0),Int32(0))
 end
 hsync(mode)
 if hc(sh,0)==0&&tid<Int32(512);hs(d.diam,tid,typemax(Int32));end
 hsync(mode)
 while hc(sh,0)<n
  source=hc(sh,1);k=tid
  while k<n
   hs(d.dist,od+k,k==source ? Int32(0) : typemax(Int32));hs(d.sigma,os+k,k==source ? UInt64(1) : UInt64(0));hs(d.delta,ode+k,0f0);k+=Int32(1024)
  end
  hsync(mode)
  if tid==0
   hs(d.Q,oq,source);hcs(sh,2,1);hcs(sh,3,0);hs(d.S,ost,source);hcs(sh,4,1);hs(d.ep,oep,Int32(0));hs(d.ep,oep+Int32(1),Int32(1));hcs(sh,6,2);hcs(sh,5,0);hcs(sh,7,0)
  end
  hsync(mode)
  r=tid+hg(d.R,source)
  while r<hg(d.R,source+Int32(1))
   w=hg(d.C,r)
   if hg(d.dist,od+w)==typemax(Int32)
    hs(d.dist,od+w,Int32(1));t=hfetch(mode,sh,Int32(3),Int32(1));hs(d.Q2,oq2+t,w)
   end
   hg(d.dist,od+w)==hg(d.dist,od+source)+Int32(1)&&hfetch(mode,d.sigma,os+w,UInt64(1));r+=Int32(1024)
  end
  hsync(mode)
  if hc(sh,3)==0
   hcs(sh,7,1)
  else
   kk=tid
   while kk<hc(sh,3);v=hg(d.Q2,oq2+kk);hs(d.Q,oq+kk,v);hs(d.S,ost+kk+hc(sh,4),v);kk+=Int32(1024);end
   hsync(mode)
   if tid==0
    le=hc(sh,6);hs(d.ep,oep+le,hg(d.ep,oep+le-Int32(1))+hc(sh,3));hcs(sh,6,le+Int32(1));hcs(sh,2,hc(sh,3));hcs(sh,4,hc(sh,4)+hc(sh,3));hcs(sh,3,0);hcs(sh,5,hc(sh,5)+Int32(1))
   end
  end
  hsync(mode)
  while hc(sh,7)==0
   if hg(d.jia,Int32(0))!=0&&hc(sh,2)>Int32(512)
    k=tid
    while k<Int32(2)*m
     v=hg(d.F,k)
     if hg(d.dist,od+v)==hc(sh,5)
      w=hg(d.C,k);nd=hg(d.dist,od+v)+Int32(1)
      if hcas(mode,d.dist,od+w,typemax(Int32),nd)==typemax(Int32);t=hfetch(mode,sh,Int32(3),Int32(1));hs(d.Q2,oq2+t,w);end
      hg(d.dist,od+w)==nd&&hfetch(mode,d.sigma,os+w,hg(d.sigma,os+v))
     end
     k+=Int32(1024)
    end
   else
    tid==0&&hcs(sh,8,1024);hsync(mode);k=tid
    while k<hc(sh,2)
     v=hg(d.Q,oq+k);r=hg(d.R,v)
     while r<hg(d.R,v+Int32(1))
      w=hg(d.C,r);nd=hg(d.dist,od+v)+Int32(1)
      if hcas(mode,d.dist,od+w,typemax(Int32),nd)==typemax(Int32);t=hfetch(mode,sh,Int32(3),Int32(1));hs(d.Q2,oq2+t,w);end
      hg(d.dist,od+w)==nd&&hfetch(mode,d.sigma,os+w,hg(d.sigma,os+v));r+=Int32(1)
     end
     k=hfetch(mode,sh,Int32(8),Int32(1))
    end
   end
   hsync(mode)
   if hc(sh,3)==0
    break
   else
    kk=tid
    while kk<hc(sh,3);v=hg(d.Q2,oq2+kk);hs(d.Q,oq+kk,v);hs(d.S,ost+kk+hc(sh,4),v);kk+=Int32(1024);end
    hsync(mode)
    if tid==0
     le=hc(sh,6);hs(d.ep,oep+le,hg(d.ep,oep+le-Int32(1))+hc(sh,3));hcs(sh,6,le+Int32(1));hcs(sh,2,hc(sh,3));hcs(sh,4,hc(sh,4)+hc(sh,3));hcs(sh,3,0);hcs(sh,5,hc(sh,5)+Int32(1))
    end
    hsync(mode)
   end
  end
  if tid==0
   hcs(sh,5,hg(d.dist,od+hg(d.S,ost+hc(sh,4)-Int32(1)))-Int32(1))
   hc(sh,0)<Int32(512)&&hs(d.diam,hc(sh,0),hc(sh,5)+Int32(1))
  end
  hsync(mode)
  while hc(sh,5)>0
   depth=hc(sh,5);len=hg(d.ep,oep+depth+Int32(1))-hg(d.ep,oep+depth)
   if hg(d.jia,Int32(0))!=0&&len>Int32(512)
    kk=tid
    while kk<Int32(2)*m
     w=hg(d.F,kk)
     if hg(d.dist,od+w)==depth
      v=hg(d.C,kk)
      if hg(d.dist,od+v)==hg(d.dist,od+w)+Int32(1)
       change=(Float32(hg(d.sigma,os+w))/Float32(hg(d.sigma,os+v)))*(1f0+hg(d.delta,ode+v));hfetch(mode,d.delta,ode+w,change)
      end
     end
     kk+=Int32(1024)
    end
   else
    kk=tid+hg(d.ep,oep+depth)
    while kk<hg(d.ep,oep+depth+Int32(1))
     w=hg(d.S,ost+kk);dsw=0f0;sw=Float32(hg(d.sigma,os+w));z=hg(d.R,w)
     while z<hg(d.R,w+Int32(1))
      v=hg(d.C,z)
      if hg(d.dist,od+v)==hg(d.dist,od+w)+Int32(1);dsw+=(sw/Float32(hg(d.sigma,os+v)))*(1f0+hg(d.delta,ode+v));end
      z+=Int32(1)
     end
     hs(d.delta,ode+w,dsw);kk+=Int32(1024)
    end
   end
   hsync(mode);tid==0&&hcs(sh,5,hc(sh,5)-Int32(1));hsync(mode)
  end
  kk=tid
  while kk<n;hfetch(mode,d.bc,kk,hg(d.delta,ode+kk));kk+=Int32(1024);end
  if tid==0
   ind=hfetch(mode,d.next,Int32(0),Int32(1));hcs(sh,0,ind);hcs(sh,1,ind)
  end
  hsync(mode)
  if hc(sh,0)==Int32(1024)
   kk=tid
   while kk<Int32(512);hcs(sh,Int32(16)+kk,hg(d.diam,kk));kk+=Int32(1024);end
   hsync(mode);bitonic!(sh,tid,mode);hsync(mode)
   if tid==0
    logn=Int32(0);temp=n>>1
    while temp!=0;logn+=Int32(1);temp>>=1;end
    hc(sh,Int32(16)+Int32(256))<Int32(4)*logn&&hs(d.jia,Int32(0),Int32(1))
   end
  end
  hsync(mode)
 end
 nothing
end
