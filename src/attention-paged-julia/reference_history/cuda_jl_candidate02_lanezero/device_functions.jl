# Translation of Apache-2.0 vLLM/FasterTransformer source functions; original
# notices and byte-identical source are retained in reference_history/.
@inline function paged_rng(x::UInt32)
 x ⊻= x<<13; x ⊻= x>>17; x ⊻= x<<5; x
end
@inline paged_uniform(x::UInt32)=Float32(x>>8)*Float32(0x1p-24)
@inline function uniform_value(i,low,high)
 u=paged_uniform(paged_rng(UInt32(1234)⊻UInt32(i)))
 muladd(high-low,u,low)
end
@inline function normal_value(i)
 r=paged_rng(UInt32(1234)⊻UInt32(i));u1=paged_uniform(r);u2=paged_uniform(paged_rng(r))
 sqrt(-2f0*log(u1))*cos(6.28318530718f0*u2)
end
@inline sx(x,m)=CUDA.shfl_xor_sync(UInt32(0xffffffff),x,Int32(m),Int32(32))
@inline sb(x)=CUDA.shfl_sync(UInt32(0xffffffff),x,Int32(0),Int32(32))
# Phase functions allow each framework to own its dispatch/shared/barriers.
@inline function load_query!(sm,q,t,h,s,H,::Val{D}) where D
 g=t÷2; off=t%2
 j=g
 @inbounds while j<D÷4
  z=off*2+j*4
  sm[off*(D÷2)+j*2+1]=q[s*H*D+h*D+z+1]
  sm[off*(D÷2)+j*2+2]=q[s*H*D+h*D+z+2]
  j+=64
 end
 nothing
end
@inline function logits!(sm,k,tables,lens,a,t,h,s,H,M,scale,::Val{D}) where D
 warp=t÷32;lane=t%32;off=t%2;g=t÷2;len=lens[s+1];blocks=cld(len,16);mx=-floatmax(Float32)
 b=warp
 @inbounds while b<blocks
  page=Int(tables[s*M+b+1]);tok=g%16;token=b*16+tok;base=page*H*D*16+h*D*16+tok*4+off*2
  qa=sm[off*(D÷2)+1];qb=sm[off*(D÷2)+2]
  aa=qa*k[base+1];bb=qb*k[base+2]
  for j in 1:D÷4-1
   aa=muladd(sm[off*(D÷2)+j*2+1],k[base+j*64+1],aa)
   bb=muladd(sm[off*(D÷2)+j*2+2],k[base+j*64+2],bb)
  end
  z=aa+bb;z=(z+sx(z,1))*scale
  slope=a[h+1];z=slope!=0f0 ? muladd(slope,Float32(token-len+1),z) : z
  if off==0
   sm[D+8+token+1]=token>=len ? 0f0 : z
   mx=token>=len ? mx : max(mx,z)
  end
  b+=4
 end
 for mask in (16,8,4,2);mx=max(mx,sx(mx,mask));end
 if lane==0;@inbounds sm[D+warp+1]=mx;end
 nothing
end
@inline function exponential!(sm,lens,t,s,::Val{D}) where D
 lane=t%32;mx=lane<4 ? sm[D+lane+1] : -floatmax(Float32)
 for mask in (2,1);mx=max(mx,sx(mx,mask));end
 mx=sb(mx);len=lens[s+1];z=0f0;i=t
 @inbounds while i<len
  e=Base.FastMath.exp_fast(sm[D+8+i+1]-mx);sm[D+8+i+1]=e;z+=e;i+=128
 end
 for mask in (16,8,4,2,1);z+=sx(z,mask);end
 if lane==0;@inbounds sm[D+4+t÷32+1]=z;end
 nothing
end
@inline function normalize!(sm,lens,t,s,::Val{D}) where D
 lane=t%32;z=lane<4 ? sm[D+4+lane+1] : 0f0
 for mask in (2,1);z+=sx(z,mask);end
 z=sb(z);inv=Base.FastMath.div_fast(1f0,z+1f-6);i=t
 @inbounds while i<lens[s+1];sm[D+8+i+1]*=inv;i+=128;end
 nothing
end
@inline function values(sm,v,tables,lens,t,h,s,H,M,::Val{D}) where D
 warp=t÷32;lane=t%32;tokenoff=(lane%4)*4;len=lens[s+1];blocks=cld(len,16)
 acc=ntuple(_->0f0,Val(D÷8));b=warp
 @inbounds while b<blocks
  page=Int(tables[s*M+b+1]);token=b*16+tokenoff
  l0=sm[D+8+token+1];l1=sm[D+8+token+2];l2=sm[D+8+token+3];l3=sm[D+8+token+4]
  acc=ntuple(Val(D÷8)) do ii
   row=lane÷4+(ii-1)*8;base=page*H*D*16+h*D*16+row*16+tokenoff
   v0=token<len ? v[base+1] : 0f0;v1=token+1<len ? v[base+2] : 0f0
   v2=token+2<len ? v[base+3] : 0f0;v3=token+3<len ? v[base+4] : 0f0
   aa=muladd(l2,v2,l0*v0);bb=muladd(l3,v3,l1*v1)
   acc[ii]+(aa+bb)
  end
  b+=4
 end
 ntuple(Val(D÷8)) do ii
  z=acc[ii];z+=sx(z,2);z+=sx(z,1);z
 end
end
@inline function store_upper!(sm,acc,t,mid,::Val{D}) where D
 warp=t÷32;lane=t%32
 if mid<=warp<2mid&&lane%4==0
  @inbounds for i in 1:D÷8;row=lane÷4+(i-1)*8;sm[D+8+(warp-mid)*D+row+1]=acc[i];end
 end
 nothing
end
@inline function add_lower(sm,acc,t,mid,::Val{D}) where D
 warp=t÷32;lane=t%32
 if warp<mid&&lane%4==0
  return ntuple(i->@inbounds(acc[i]+sm[D+8+warp*D+lane÷4+(i-1)*8+1]),Val(D÷8))
 end
 acc
end
@inline function finish_output!(out,acc,t,h,s,H,::Val{D}) where D
 if t÷32==0&&t%4==0
  @inbounds for i in 1:D÷8;row=(t%32)÷4+(i-1)*8;out[s*H*D+h*D+row+1]=acc[i];end
 end
 nothing
end
