# Source TwoOpt algorithm, BSD notice retained in LICENSE.
@inline pxget(g,base,n,i)=reinterpret(Float32,@inbounds(g[base+n+i+1]))
@inline pyget(g,base,n,i)=reinterpret(Float32,@inbounds(g[base+2n+i+2]))
@inline function pxset!(g,base,n,i,x)
 @inbounds g[base+n+i+1]=reinterpret(Int32,x)
end
@inline function pyset!(g,base,n,i,x)
 @inbounds g[base+2n+i+2]=reinterpret(Int32,x)
end
@inline function distance(x0,y0,x1,y1)
 dx=x0-x1;dy=y0-y1
 trunc(Int32,sqrt(dx*dx+dy*dy))
end
@inline distg(g,base,n,i,j)=distance(pxget(g,base,n,i),pyget(g,base,n,i),pxget(g,base,n,j),pyget(g,base,n,j))
function twoopt_body!(n,x,y,g,climbs,best,sh,tid,bid,t)
 stride=((3n+2+31)÷32)*32;base=bid*stride
 for i in tid:t:n-1
  @inbounds pxset!(g,base,n,i,x[i+1])
  @inbounds pyset!(g,base,n,i,y[i+1])
 end
 CUDA.sync_threads()
 if tid==0
  seed=UInt32(bid)
  for i in 1:n-1
   seed=(UInt32(26757677)*seed+UInt32(1))&UInt32(0x7fffffff)
   r=Float32(seed)/Float32(2147483648)
   j=trunc(Int, r*Float32(n-1))+1
   a=pxget(g,base,n,i);b=pxget(g,base,n,j);pxset!(g,base,n,i,b);pxset!(g,base,n,j,a)
   a=pyget(g,base,n,i);b=pyget(g,base,n,j);pyset!(g,base,n,i,b);pyset!(g,base,n,j,a)
  end
  pxset!(g,base,n,n,pxget(g,base,n,0));pyset!(g,base,n,n,pyget(g,base,n,0))
 end
 CUDA.sync_threads()
 while true
  for i in tid:t:n-1;@inbounds g[base+i+1]=-distg(g,base,n,i,i+1);end
  CUDA.sync_threads()
  minchange=Int32(0);mini=1;minj=0
  for ii in 0:t:n-3
   i=ii+tid;pxi0=0f0;pyi0=0f0;pxi1=0f0;pyi1=0f0;pxj1=0f0;pyj1=0f0
   if i<n-2
    @inbounds minchange-=g[base+i+1]
    pxi0=pxget(g,base,n,i);pyi0=pyget(g,base,n,i);pxi1=pxget(g,base,n,i+1);pyi1=pyget(g,base,n,i+1);pxj1=pxget(g,base,n,n);pyj1=pyget(g,base,n,n)
   end
   for jj in n-1:-128:ii+2
    bound=jj-127
    for k in tid:t:127
     if k+bound>=ii+2
      @inbounds sh[t+k+1]=reinterpret(Int32,pxget(g,base,n,k+bound))
      @inbounds sh[t+128+k+1]=reinterpret(Int32,pyget(g,base,n,k+bound))
      @inbounds sh[t+256+k+1]=g[base+k+bound+1]
     end
    end
    CUDA.sync_threads()
    lower=max(bound,i+2)
    for j in jj:-1:lower
     jm=j-bound
     @inbounds pxj0=reinterpret(Float32,sh[t+jm+1])
     @inbounds pyj0=reinterpret(Float32,sh[t+128+jm+1])
     @inbounds change=sh[t+256+jm+1]+distance(pxi0,pyi0,pxj0,pyj0)+distance(pxi1,pyi1,pxj1,pyj1)
     pxj1=pxj0;pyj1=pyj0
     if minchange>change;minchange=change;mini=i;minj=j;end
    end
    CUDA.sync_threads()
   end
   if i<n-2;@inbounds minchange+=g[base+i+1];end
  end
  CUDA.sync_threads()
  change=minchange;@inbounds sh[tid+1]=minchange
  if tid==0;CUDA.atomic_add!(pointer(climbs),Int32(1));end
  CUDA.sync_threads()
  j=t
  while true
   k=(j+1)÷2
   if tid+k<j
    @inbounds tmp=sh[tid+k+1]
    if change>tmp;change=tmp;end
    @inbounds sh[tid+1]=change
   end
   j=k;CUDA.sync_threads()
   j<=1&&break
  end
  @inbounds if minchange==sh[1];sh[2]=Int32(tid);end # source nondeterministic tie winner preserved
  CUDA.sync_threads()
  @inbounds if tid==sh[2];sh[3]=Int32(mini+1);sh[4]=Int32(minj);end
  CUDA.sync_threads()
  @inbounds minchange=sh[1]
  @inbounds first=Int(sh[3]);last=Int(sh[4]);ss=last+first
  i=tid
  while i+i<ss
   if first<=i
    j=ss-i;a=pxget(g,base,n,i);b=pxget(g,base,n,j);pxset!(g,base,n,i,b);pxset!(g,base,n,j,a)
    a=pyget(g,base,n,i);b=pyget(g,base,n,j);pyset!(g,base,n,i,b);pyset!(g,base,n,j,a)
   end
   i+=t
  end
  CUDA.sync_threads()
  minchange>=0&&break
 end
 term=Int32(0)
 for i in tid:t:n-1;term+=distg(g,base,n,i,i+1);end
 @inbounds sh[tid+1]=term
 CUDA.sync_threads();j=t
 while true
  k=(j+1)÷2
  if tid+k<j;@inbounds term+=sh[tid+k+1];end
  CUDA.sync_threads()
  if tid+k<j;@inbounds sh[tid+1]=term;end
  j=k;CUDA.sync_threads();j<=1&&break
 end
 if tid==0;CUDA.atomic_min!(pointer(best),term);end
 return
end
