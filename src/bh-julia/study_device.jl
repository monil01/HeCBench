# ECL-BH4.5 mathematical pipeline. Zero-based helpers mirror native indexing.
# CUDA backend synchronization/atomics/warp votes are explicit backend extensions.
@inline bg(a,i)=@inbounds a[i+Int32(1)]
@inline bs(a,i,x)=(@inbounds a[i+Int32(1)]=x)
@inline bp(p,i,c)=bg(p,Int32(4)*i+Int32(c))
@inline bps(p,i,c,x)=bs(p,Int32(4)*i+Int32(c),x)
@inline bz()=threadIdx().x-Int32(1)
@inline bb()=blockIdx().x-Int32(1)
@inline bi()=bz()+bb()*blockDim().x
@inline binc()=blockDim().x*gridDim().x

# Explicit LLVM volatile scalar loads/stores. Volatile is not an atomic or fence.
@generated function bvload(p::Core.LLVMPtr{T,A}) where {T,A}
 ty=T===Float32 ? "float" : T===Int32 ? "i32" : error("BH volatile type")
 ir="define $ty @entry(i64 %x) { %p = inttoptr i64 %x to ptr addrspace($A)\n %v = load volatile $ty, ptr addrspace($A) %p, align 4\n ret $ty %v }"
 :(Base.llvmcall(($ir,"entry"),T,Tuple{UInt},UInt(p)))
end
@generated function bvstore(p::Core.LLVMPtr{T,A},x::T) where {T,A}
 ty=T===Float32 ? "float" : T===Int32 ? "i32" : error("BH volatile type")
 ir="define void @entry(i64 %x, $ty %v) { %p = inttoptr i64 %x to ptr addrspace($A)\n store volatile $ty %v, ptr addrspace($A) %p, align 4\n ret void }"
 :(Base.llvmcall(($ir,"entry"),Cvoid,Tuple{UInt,T},UInt(p),x))
end
@inline bvg(a,i)=bvload(pointer(a,i+Int32(1)))
@inline bvs(a,i,x)=bvstore(pointer(a,i+Int32(1)),x)
@inline bvp(p,i,c)=bvg(p,Int32(4)*i+Int32(c))
@inline bvps(p,i,c,x)=bvs(p,Int32(4)*i+Int32(c),x)

function bh_init!(step,bc)
 bs(step,Int32(0),Int32(-1));bs(bc,Int32(0),Int32(0));return
end
function bh_bbox!(nn,nb,start,child,p,maxs,mins,radius,bottom,step,bc)
 sh=CuStaticSharedArray(Float32,6*256);t=bz();i=bi();inc=binc()
 xmin=bp(p,Int32(0),0);xmax=xmin;ymin=bp(p,Int32(0),1);ymax=ymin;zmin=bp(p,Int32(0),2);zmax=zmin
 while i<nb
  x=bp(p,i,0);y=bp(p,i,1);z=bp(p,i,2)
  xmin=min(xmin,x);xmax=max(xmax,x);ymin=min(ymin,y);ymax=max(ymax,y);zmin=min(zmin,z);zmax=max(zmax,z);i+=inc
 end
 bs(sh,t,xmin);bs(sh,t+Int32(256),xmax);bs(sh,t+Int32(512),ymin);bs(sh,t+Int32(768),ymax);bs(sh,t+Int32(1024),zmin);bs(sh,t+Int32(1280),zmax)
 j=Int32(128)
 while j>0
  sync_threads()
  if t<j
   k=t+j;xmin=min(xmin,bg(sh,k));xmax=max(xmax,bg(sh,k+Int32(256)));ymin=min(ymin,bg(sh,k+Int32(512)));ymax=max(ymax,bg(sh,k+Int32(768)));zmin=min(zmin,bg(sh,k+Int32(1024)));zmax=max(zmax,bg(sh,k+Int32(1280)))
   bs(sh,t,xmin);bs(sh,t+Int32(256),xmax);bs(sh,t+Int32(512),ymin);bs(sh,t+Int32(768),ymax);bs(sh,t+Int32(1024),zmin);bs(sh,t+Int32(1280),zmax)
  end
  j÷=Int32(2)
 end
 if t==0
  k=Int32(3)*bb();bs(mins,k,xmin);bs(mins,k+Int32(1),ymin);bs(mins,k+Int32(2),zmin);bs(maxs,k,xmax);bs(maxs,k+Int32(1),ymax);bs(maxs,k+Int32(2),zmax)
  threadfence();last=Int32(gridDim().x-Int32(1))
  if CUDA.atomic_inc!(pointer(bc),last)==last
   for j in Int32(0):Int32(last)
    k=Int32(3)*j;xmin=min(xmin,bg(mins,k));xmax=max(xmax,bg(maxs,k));ymin=min(ymin,bg(mins,k+Int32(1)));ymax=max(ymax,bg(maxs,k+Int32(1)));zmin=min(zmin,bg(mins,k+Int32(2)));zmax=max(zmax,bg(maxs,k+Int32(2)))
   end
   bs(radius,Int32(0),max(max(xmax-xmin,ymax-ymin),zmax-zmin)*0.5f0);bs(bottom,Int32(0),nn);bs(start,nn,Int32(0))
   bps(p,nn,0,(xmin+xmax)*0.5f0);bps(p,nn,1,(ymin+ymax)*0.5f0);bps(p,nn,2,(zmin+zmax)*0.5f0);bps(p,nn,3,-1f0)
   for j in Int32(0):Int32(7);bs(child,Int32(8)*nn+j,Int32(-1));end
   bs(step,Int32(0),bg(step,Int32(0))+Int32(1))
  end
 end
 return
end
function bh_clear1!(nn,nb,child)
 lo=Int32(8)*nb;k=(lo & Int32(-32))+bi();inc=binc();k<lo&&(k+=inc)
 while k<Int32(8)*nn;bs(child,k,Int32(-1));k+=inc;end
 return
end
function bh_build!(nn,nb,child,p,radius,bottom)
 radius0=bg(radius,Int32(0))*0.5f0;rx=bp(p,nn,0);ry=bp(p,nn,1);rz=bp(p,nn,2)
 skip=Int32(1);inc=binc();i=bi();n=Int32(0);j=Int32(0);r=0f0;x=0f0;y=0f0;z=0f0;locked=Int32(0);patch=Int32(0)
 while i<nb
  px=bp(p,i,0);py=bp(p,i,1);pz=bp(p,i,2)
  if skip!=0
   skip=Int32(0);n=nn;r=radius0;dx=-r;dy=-r;dz=-r;j=Int32(0)
   if rx<px;j=Int32(1);dx=r;end
   if ry<py;j|=Int32(2);dy=r;end
   if rz<pz;j|=Int32(4);dz=r;end
   x=rx+dx;y=ry+dy;z=rz+dz
  end
  ch=bvg(child,n*Int32(8)+j)
  while ch>=nb
   n=ch;r*=0.5f0;dx=-r;dy=-r;dz=-r;j=Int32(0)
   if x<px;j=Int32(1);dx=r;end
   if y<py;j|=Int32(2);dy=r;end
   if z<pz;j|=Int32(4);dz=r;end
   x+=dx;y+=dy;z+=dz;ch=bvg(child,n*Int32(8)+j)
  end
  if ch!=Int32(-2)
   locked=n*Int32(8)+j
   if ch==Int32(-1)
    if CUDA.atomic_cas!(pointer(child,locked+Int32(1)),ch,i)==ch;i+=inc;skip=Int32(1);end
   elseif CUDA.atomic_cas!(pointer(child,locked+Int32(1)),ch,Int32(-2))==ch
    patch=Int32(-1);cx=bp(p,ch,0);cy=bp(p,ch,1);cz=bp(p,ch,2)
    while true
     cell=CUDA.atomic_sub!(pointer(bottom),Int32(1))-Int32(1)
     if patch!=Int32(-1);bvs(child,n*Int32(8)+j,cell);end
     patch=max(patch,cell);j=Int32(0)
     x<cx&&(j=Int32(1));y<cy&&(j|=Int32(2));z<cz&&(j|=Int32(4));bvs(child,cell*Int32(8)+j,ch)
     n=cell;r*=0.5f0;dx=-r;dy=-r;dz=-r;j=Int32(0)
     if x<px;j=Int32(1);dx=r;end
     if y<py;j|=Int32(2);dy=r;end
     if z<pz;j|=Int32(4);dz=r;end
     x+=dx;y+=dy;z+=dz;ch=bvg(child,n*Int32(8)+j)
     ch<0&&break
    end
    bvs(child,n*Int32(8)+j,i);i+=inc;skip=Int32(2)
   end
  end
  sync_threads()
  if skip==Int32(2);bvs(child,locked,patch);end
 end
 return
end
function bh_clear2!(nn,start,p,bottom)
 lo=bg(bottom,Int32(0));inc=binc();k=(lo & Int32(-32))+bi();k<lo&&(k+=inc)
 while k<nn;bps(p,k,3,-1f0);bs(start,k,Int32(-1));k+=inc;end
 return
end
@inline function bh_sum_ready!(k,nb,count,p,childcache,masscache)
 cm=0f0;px=0f0;py=0f0;pz=0f0;cnt=Int32(0);t=bz()
 for i in Int32(0):Int32(7)
  ch=bg(childcache,i*Int32(256)+t)
  if ch>=0
   x=bvp(p,ch,0);y=bvp(p,ch,1);z=bvp(p,ch,2);w=bvp(p,ch,3)
   if ch>=nb;m=bg(masscache,i*Int32(256)+t);cnt+=bvg(count,ch);else;m=w;cnt+=Int32(1);end
   cm+=m;px=muladd(x,m,px);py=muladd(y,m,py);pz=muladd(z,m,pz)
  end
 end
 bvs(count,k,cnt);m=1f0/cm;bvps(p,k,0,px*m);bvps(p,k,1,py*m);bvps(p,k,2,pz*m);bvps(p,k,3,cm);return
end
function bh_summarize!(nn,nb,count,child,p,bottom)
 cc=CuStaticSharedArray(Int32,256*8);mc=CuStaticSharedArray(Float32,256*8)
 lo=bg(bottom,Int32(0));inc=binc();k=(lo & Int32(-32))+bi();k<lo&&(k+=inc);restart=k;t=bz()
 for pass in Int32(0):Int32(2)
  while k<=nn
   if bvp(p,k,3)<0f0
    ready=true
    for i in Int32(0):Int32(7)
     ch=bg(child,k*Int32(8)+i);s=i*Int32(256)+t;bs(cc,s,ch)
     if ch>=nb
      m=bvp(p,ch,3);bs(mc,s,m)
      if m<0f0;ready=false;break;end
     end
    end
    ready&&bh_sum_ready!(k,nb,count,p,cc,mc)
   end
   k+=inc
  end
  k=restart
 end
 j=Int32(0)
 while k<=nn
  if bvp(p,k,3)>=0f0;k+=inc
  else
   if j==0
    j=Int32(8)
    for i in Int32(0):Int32(7)
     ch=bg(child,k*Int32(8)+i);s=i*Int32(256)+t;bs(cc,s,ch)
     if ch<nb;j-=Int32(1)
     else;m=bvp(p,ch,3);bs(mc,s,m);m>=0f0&&(j-=Int32(1));end
    end
   else
    j=Int32(8)
    for i in Int32(0):Int32(7)
     s=i*Int32(256)+t;ch=bg(cc,s)
     if ch<nb || bg(mc,s)>=0f0;j-=Int32(1)
     else;m=bvp(p,ch,3);bs(mc,s,m);m>=0f0&&(j-=Int32(1));end
    end
   end
   if j==0;bh_sum_ready!(k,nb,count,p,cc,mc);k+=inc;end
  end
 end
 return
end
function bh_sort!(nn,nb,sort,count,start,child,bottom)
 lo=bg(bottom,Int32(0));dec=binc();k=nn+Int32(1)-dec+bi()
 while k>=lo
  s=bvg(start,k)
  if s>=0
   j=Int32(0)
   for i in Int32(0):Int32(7)
    ch=bg(child,k*Int32(8)+i)
    if ch>=0
     if i!=j;bs(child,k*Int32(8)+i,Int32(-1));bs(child,k*Int32(8)+j,ch);end
     j+=Int32(1)
     if ch>=nb;bvs(start,ch,s);s+=bg(count,ch)
     else;bs(sort,s,ch);s+=Int32(1);end
    end
   end
   k-=dec
  end
  sync_threads()
 end
 return
end
function bh_force!(nn,nb,halfdt,itolsq,epssq,sort,child,p,vel,acc,radius,step)
 pos=CuStaticSharedArray(Int32,256);node=CuStaticSharedArray(Int32,256);dq=CuStaticSharedArray(Float32,256);t=bz()
 if t==0
  tmp=bg(radius,Int32(0))*2f0;bs(dq,Int32(0),tmp*tmp*itolsq)
  for i in Int32(1):Int32(31);bs(dq,i,bg(dq,i-Int32(1))*0.25f0);bs(dq,i-Int32(1),bg(dq,i-Int32(1))+epssq);end
  bs(dq,Int32(31),bg(dq,Int32(31))+epssq)
 end
 sync_threads();base=t÷Int32(32);sbase=base*Int32(32);j=base*Int32(32);diff=t-sbase
 if base>0 && diff<Int32(32);bs(dq,diff+j,bg(dq,diff));end
 sync_threads();k=bi();inc=binc()
 while k<nb
  i=bg(sort,k);px=bp(p,i,0);py=bp(p,i,1);pz=bp(p,i,2);ax=0f0;ay=0f0;az=0f0;depth=j
  if sbase==t;bs(pos,j,Int32(0));bs(node,j,nn*Int32(8));end
  sync_warp()
  while true
   pd=bg(pos,depth);nd=bg(node,depth);sync_warp()
   while pd<Int32(8)
    n=bg(child,nd+pd);pd+=Int32(1)
    if n>=0
     dx=bp(p,n,0)-px;dy=bp(p,n,1)-py;dz=bp(p,n,2)-pz
     tmp=muladd(dx,dx,muladd(dy,dy,muladd(dz,dz,epssq)))
     if n<nb || CUDA.vote_all_sync(UInt32(0xffffffff),tmp>=bg(dq,depth))
      tmp=CUDA.rsqrt(tmp);tmp=bp(p,n,3)*tmp*tmp*tmp
      ax=muladd(dx,tmp,ax);ay=muladd(dy,tmp,ay);az=muladd(dz,tmp,az)
     else
      if sbase==t;bs(pos,depth,pd);bs(node,depth,nd);end
      depth+=Int32(1);pd=Int32(0);nd=n*Int32(8)
     end
    else;pd=Int32(8);end
   end
   depth-=Int32(1);sync_warp();depth<j&&break
  end
  if bg(step,Int32(0))>0
   for (c,a) in ((Int32(0),ax),(Int32(1),ay));v=bg(vel,Int32(2)*i+c);v=muladd(a-bp(acc,i,c),halfdt,v);bs(vel,Int32(2)*i+c,v);end
   bps(acc,i,3,muladd(az-bp(acc,i,2),halfdt,bp(acc,i,3)))
  end
  bps(acc,i,0,ax);bps(acc,i,1,ay);bps(acc,i,2,az);k+=inc
 end
 return
end
function bh_integrate!(nb,dt,halfdt,p,vel,acc)
 i=bi();inc=binc()
 while i<nb
  dx=bp(acc,i,0)*halfdt;dy=bp(acc,i,1)*halfdt;dz=bp(acc,i,2)*halfdt
  hx=bg(vel,Int32(2)*i)+dx;hy=bg(vel,Int32(2)*i+Int32(1))+dy;hz=bp(acc,i,3)+dz
  bps(p,i,0,muladd(hx,dt,bp(p,i,0)));bps(p,i,1,muladd(hy,dt,bp(p,i,1)));bps(p,i,2,muladd(hz,dt,bp(p,i,2)))
  bs(vel,Int32(2)*i,hx+dx);bs(vel,Int32(2)*i+Int32(1),hy+dy);bps(acc,i,3,hz+dz);i+=inc
 end
 return
end
