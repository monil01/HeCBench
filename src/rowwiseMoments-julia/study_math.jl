struct WData
 mean::Float32
 m2::Float32
 n::Int64
 nf::Float32
end
const WZERO=WData(0f0,0f0,Int64(0),0f0)
@inline function wreduce(a::WData,x::Float32)
 n=a.n+1;nf=Float32(n);delta=x-a.mean;mean=a.mean+delta/nf;nd=x-mean
 return WData(mean,muladd(delta,nd,a.m2),n,nf)
end
@inline function wcombine(a::WData,b::WData)
 if a.nf==0f0;return b;end
 if b.nf==0f0;return a;end
 delta=b.mean-a.mean;cnt=a.nf+b.nf;ratio=b.nf/cnt
 return WData(muladd(delta,ratio,a.mean),muladd((delta*delta)*a.nf,ratio,a.m2+b.m2),Int64(-1),cnt)
end
@inline function wshuffle(a::WData,d)
 mask=UInt32(0xffffffff)
 return WData(CUDA.shfl_down_sync(mask,a.mean,d),CUDA.shfl_down_sync(mask,a.m2,d),CUDA.shfl_down_sync(mask,a.n,d),CUDA.shfl_down_sync(mask,a.nf,d))
end
@inline function warp_reduce(a::WData)
 for d in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1));a=wcombine(a,wshuffle(a,d));end
 return a
end
@inline function moments_body(tid,row,n,eps,x,mean,rstd,shared)
 a=WZERO
 j=Int64(tid)
 while j<n
  @inbounds a=wreduce(a,x[(Int64(row)-1)*n+j+1]);j+=256
 end
 a=warp_reduce(a);CUDA.sync_threads();lane=tid%32;warp=tid÷32
 if lane==0;@inbounds shared[warp+1]=a;end
 CUDA.sync_threads()
 a=tid<8 ? (@inbounds shared[lane+1]) : WZERO
 if warp==0;a=warp_reduce(a);end
 if tid==0
  @inbounds begin
   rstd[row]=CUDA.rsqrt(a.m2/a.nf+eps)
   mean[row]=a.mean
  end
 end
 return
end
