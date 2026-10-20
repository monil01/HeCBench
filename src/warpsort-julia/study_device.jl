# Faithful source register-bitonic/recursive-merge translation; no vendor sorter.
struct SortPair
 k::Float32
 v::Int32
end
@inline greater(a::Float32,b::Float32)=a>b
@inline greater(a::SortPair,b::SortPair)=a.k>b.k || (a.k==b.k && a.v>b.v)
for fn in (:shfl_sync,:shfl_up_sync,:shfl_down_sync,:shfl_xor_sync)
 @eval begin
  @inline $(Symbol(:ws_,fn))(x::Float32,i)=CUDA.$fn(typemax(UInt32),x,i)
  @inline $(Symbol(:ws_,fn))(x::SortPair,i)=SortPair(CUDA.$fn(typemax(UInt32),x.k,i),CUDA.$fn(typemax(UInt32),x.v,i))
 end
end
@inline function swapstep(x,mask,dir)
 y=ws_shfl_xor_sync(x,mask)
 return greater(x,y)==dir ? y : x
end
@inline function bitonic(x,lane)
 for level in 1:4
  for p in (level-1):-1:0
   x=swapstep(x,Int32(1)<<p,((lane>>level)&1)⊻((lane>>p)&1)!=0)
  end
 end
 for p in 4:-1:0
  x=swapstep(x,Int32(1)<<p,((lane>>p)&1)!=0)
 end
 return x
end
@inline function getmulti(a,index,val,lane)
 load=a[(index>>5)+1]
 shift=ws_shfl_down_sync(load,Int32(16))
 load=((index>>4)&1)!=0 ? shift : load
 return lane<16 ? load : val
end
@inline function broadcastreg(a,index)
 return ws_shfl_sync(a[(index>>5)+1],(index&31)+Int32(1)) # CUDA.jl lanes are1based
end
@inline function warpmrg(a::NTuple{M,T},b::NTuple{N,T},lane) where {M,N,T}
 val=ws_shfl_up_sync(b[1],Int32(16));val=lane<16 ? a[1] : val
 ai=Int32(16);bi=Int32(16);di=Int32(0)
 dst=ntuple(_->a[1],Val(M+N)) # immutable capture avoids Core.Box of loop-mutated val
 while true
  val=bitonic(val,lane)
  if di < (M+N)*32-32
   half=(di>>4)&1
   shift=ws_shfl_up_sync(val,Int32(16));vv=half!=0 ? shift : val
   if (lane & (half<<4)) == (half<<4)
    dst=Base.setindex(dst,vv,(di>>5)+1)
   end
   di+=Int32(16)
  else
   dst=Base.setindex(dst,val,M+N);break
  end
  if ai==32*M
   val=getmulti(b,bi,val,lane);bi+=Int32(16)
  elseif bi==32*N
   val=getmulti(a,ai,val,lane);ai+=Int32(16)
  else
   ca=broadcastreg(a,ai-Int32(1));cb=broadcastreg(b,bi-Int32(1))
   if greater(ca,cb)
    val=getmulti(a,ai,val,lane);ai+=Int32(16)
   else
    val=getmulti(b,bi,val,lane);bi+=Int32(16)
   end
  end
 end
 return dst
end
@inline splitmerge(a::NTuple{1,T},lane) where T=a
@inline function splitmerge(a::NTuple{N,T},lane) where {N,T}
 m=N÷2
 left=ntuple(i->a[i],Val(m));right=ntuple(i->a[i+m],Val(N-m))
 return warpmrg(splitmerge(left,lane),splitmerge(right,lane),lane)
end
@inline function sortregs(inp,out,ix,n,lane,::Val{N},::Val{PAIR}) where {N,PAIR}
 regs=ntuple(Val(N)) do i
  offset=lane+Int32((i-1)*32)
  k=offset<n ? @inbounds(inp[offset+1]) : -Inf32
  PAIR ? SortPair(k,offset<n ? offset : typemin(Int32)) : k
 end
 sorted=splitmerge(ntuple(i->bitonic(regs[i],lane),Val(N)),lane)
 for i in 1:N
  offset=lane+Int32((i-1)*32)
  if offset<n
   x=sorted[i]
   if PAIR
    @inbounds out[offset+1]=x.k
    @inbounds ix[offset+1]=x.v
   else
    @inbounds out[offset+1]=x
   end
  end
 end
 return
end
@inline function sortbody(inp,out,ix,n,lane,mode)
 if n<=32
  sortregs(inp,out,ix,n,lane,Val(1),mode)
 elseif n<=64
  sortregs(inp,out,ix,n,lane,Val(2),mode)
 elseif n<=96
  sortregs(inp,out,ix,n,lane,Val(3),mode)
 else
  sortregs(inp,out,ix,n,lane,Val(4),mode)
 end
 return
end
