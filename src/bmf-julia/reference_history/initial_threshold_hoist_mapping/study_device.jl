@inline function bhash(x::UInt32)
 x=(x+0x7ed55d16)+(x<<12);x=(x⊻0xc761c23c)⊻(x>>19)
 x=(x+0x165667b1)+(x<<5);x=(x+0xd3a2646c)⊻(x<<9)
 x=(x+0xfd7046c5)+(x<<3);(x⊻0xb55a4f09)⊻(x>>16)
end
@inline function rnginit(seed::UInt32)
 w=iszero(seed) ? typemax(UInt32) : seed;x=y=z=w
 for _ in 1:8;x=bhash(w);y=bhash(x);z=bhash(y);w=bhash(z);end
 (x,y,z,w,UInt32(0))
end
@inline function rngnext(s)
 x,y,z,w,c=s;y⊻=y<<5;y⊻=y>>7;y⊻=y<<22;t=z+w+c
 x+=UInt32(1411392427);ww=t&0x7fffffff
 (x+y+ww,(x,y,w,ww,UInt32(t>=0x80000000)))
end
@inline function flipmask(s,dim,chance,depth)
 r,s=rngnext(s)
 if Float32(r)/Float32(typemax(UInt32))<chance
  m=typemax(UInt32)>>(32-dim)
  for _ in 1:depth;r,s=rngnext(s);m&=r;end
  m
 else
  r,s=rngnext(s);UInt32(1)<<(r%UInt32(dim))
 end
end
@inline function metropolis(s,error,temp,n)
 error<=0f0 && return true
 temp<=0f0 && return false
 r,_=rngnext(s);Float32(r)/Float32(typemax(UInt32))<exp((-error/Float32(n))/temp)
end
@inline function warpsum(v,width=32)
 off=width÷2
 while off>0;v+=CUDA.shfl_down_sync(typemax(UInt32),v,off);off÷=2;end
 v
end
@inline measure(prod,t,weight)=t==1 ? weight*Float32(prod⊻t) : Float32(prod⊻t)
@inline function initbody(A,n,dim,seed,rd,tid,bid)
 i=tid+bid*512
 if i<n
  s=rnginit(seed+UInt32(i));v=UInt32(0)
  if rd<16
   v=typemax(UInt32)>>(32-dim)
   for _ in 1:rd;r,s=rngnext(s);v&=r;end
  end
  @inbounds A[i+1]=v
 end
 nothing
end
@inline function distbody(A,B,C,h,w,pw,weight,E,tid,bid)
 bs=CUDA.CuStaticSharedArray(UInt32,512);cs=CUDA.CuStaticSharedArray(UInt32,512)
 sums=CUDA.CuStaticSharedArray(Float32,16)
 lane=tid%32;wid=tid÷32;i=bid*16+wid;row=i÷32;bit=i%32
 ai=i<h ? @inbounds(A[i+1]) : UInt32(0);err=0f0
 for start in 0:512:(cld(w,512)*512-1)
  j=start+tid
  @inbounds bs[tid+1]=j<w ? B[j+1] : UInt32(0)
  @inbounds cs[tid+1]=j<w ? C[row*pw+j+1] : UInt32(0)
  CUDA.sync_threads()
  if i<h
   for k in 0:15
    @inbounds b=bs[k*32+lane+1];@inbounds t=Int32((cs[k*32+lane+1]>>bit)&UInt32(1))
    err+=measure(Int32((ai&b)!=0),t,weight)
   end
  end
  CUDA.sync_threads()
 end
 err=warpsum(err)
 if lane==0;@inbounds sums[wid+1]=err;end
 CUDA.sync_threads()
 if wid==0
  val=lane<16 ? @inbounds(sums[lane+1]) : 0f0;val=warpsum(val,16)
  if tid==0;CUDA.atomic_add!(pointer(E,1),val);end
 end
 nothing
end
@inline function updatebody(A,B,C,h,w,pw,dim,start,E,seed,temp,chance,depth,weight,tid,bid,::Val{COL}) where COL
 sharedf=CUDA.CuStaticSharedArray(UInt32,512);sharedc=CUDA.CuStaticSharedArray(UInt32,512)
 lane=tid%32;warp=bid*16+tid÷32;n=COL ? w : h;other=COL ? h : w
 at=(start+warp)%(cld(n,16)*16);target=COL ? B : A;opposite=COL ? A : B
 old=at<n ? @inbounds(target[at+1]) : UInt32(0);s=rnginit(seed+UInt32(warp));new=at<n ? old⊻flipmask(s,dim,chance,depth) : UInt32(0);err=0f0
 for base in 0:512:(cld(other,512)*512-1)
  j=base+tid
  @inbounds sharedf[tid+1]=j<other ? opposite[j+1] : UInt32(0)
  if COL
   @inbounds sharedc[tid+1]=(j÷32)<cld(h,32) ? C[(j÷32)*pw+(at÷32)*32+lane+1] : UInt32(0)
  else
   @inbounds sharedc[tid+1]=j<w ? C[(at÷32)*pw+j+1] : UInt32(0)
  end
  CUDA.sync_threads()
  if at<n
   for k in 0:15
    @inbounds b=sharedf[k*32+lane+1]
    index=COL ? k*32+at%32+1 : k*32+lane+1;shift=COL ? lane : at%32
    @inbounds truth=Int32((sharedc[index]>>shift)&UInt32(1))
    err+=measure(Int32((new&b)!=0),truth,weight)-measure(Int32((old&b)!=0),truth,weight)
   end
  end
  CUDA.sync_threads()
 end
 if at<n
  err=warpsum(err)
  if lane==0 && metropolis(s,err,temp,other)
   @inbounds target[at+1]=new;CUDA.atomic_add!(pointer(E,1),err)
  end
 end
 nothing
end
