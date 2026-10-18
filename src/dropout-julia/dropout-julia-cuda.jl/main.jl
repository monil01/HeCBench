using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
function dropout_scalar!(a,b,mask,n,p,seed,offset)
 idx=UInt32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);threads=UInt32(gridDim().x*blockDim().x);stride=threads*UInt32(4);rounded=((n-UInt32(1))÷stride+UInt32(1))*stride;linear=idx;scale=1f0/p;counter,cache=initial_rng(seed,UInt64(idx),offset)
 while linear<rounded
  counter,cache,raw=rng_next(counter,cache,seed,offset);keep=let r=raw,prob=p;ntuple(i->rng_uniform(r[i])<prob,Val(4));end
  src=let pos=linear,t=threads,total=n,arr=a;ntuple(i->(@inbounds pos+t*UInt32(i-1)<total ? arr[pos+t*UInt32(i-1)+1] : 0f0),Val(4));end
  @inbounds for lane in 0:3
   li=linear+threads*UInt32(lane)
   if li<n;b[li+1]=src[lane+1]*Float32(keep[lane+1])*scale;mask[li+1]=UInt8(keep[lane+1]);end
  end
  sync_threads();linear+=stride
 end
 return
end
function dropout_vector!(::Val{V},a,b,mask,n,p,seed,offset) where V
 idx=UInt32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);block0=UInt32(blockIdx().x-1);threads=UInt32(gridDim().x*blockDim().x);stride=threads*UInt32(V);linear=idx*UInt32(V);first=block0*UInt32(256*V);loops=first<n ? (n-UInt32(1)-first)÷stride+UInt32(1) : UInt32(0);scale=1f0/p;counter,cache=initial_rng(seed,UInt64(idx),offset)
 for iteration in UInt32(1):loops
  if linear<n
   counter,cache,raw=rng_next(counter,cache,seed,offset);keep=let r=raw,prob=p;ntuple(i->rng_uniform(r[i])<prob,Val(V));end;index=linear÷UInt32(V)+UInt32(1);src=@inbounds a[index]
   result=let values=src,bits=keep,s=scale;ntuple(i->VecElement(values[i].value*Float32(bits[i])*s),Val(V));end
   bits=let flags=keep;ntuple(i->VecElement(UInt8(flags[i])),Val(V));end
   @inbounds b[index]=result;@inbounds mask[index]=bits
  end
  sync_threads();linear+=stride
 end
 return
end
function rng_probe_kernel!(out,subseqs,seed,offset)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=5
  counter,cache=initial_rng(seed,@inbounds(subseqs[i]),offset)
  @inbounds for iteration in 0:3
   counter,cache,raw=rng_next(counter,cache,seed,offset)
   for lane in 1:4;out[(i-1)*16+iteration*4+lane]=raw[lane];end
  end
 end
 return
end
function launch(vec,a,b,mask,n,p,seed,offset)
 if vec==1
  @cuda threads=256 blocks=512 dropout_scalar!(a,b,mask,n,p,seed,offset)
 else
  @cuda threads=256 blocks=(vec==2 ? 256 : 128) dropout_vector!(Val(vec),a,b,mask,n,p,seed,offset)
 end
end
probe(out,seq,seed,offset)=(@cuda threads=32 blocks=1 rng_probe_kernel!(out,seq,seed,offset))
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch,probe)
