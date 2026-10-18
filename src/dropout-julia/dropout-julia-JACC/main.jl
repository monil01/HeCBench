using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function dropout_scalar!(global_idx,a,b,mask,n,p,seed,offset,total_threads)
 idx=UInt32(global_idx-1);threads=total_threads;stride=threads*UInt32(4);rounded=((n-UInt32(1))÷stride+UInt32(1))*stride;linear=idx;scale=1f0/p;counter,cache=initial_rng(seed,UInt64(idx),offset)
 while linear<rounded
  counter,cache,raw=rng_next(counter,cache,seed,offset);keep=let r=raw,prob=p;ntuple(i->rng_uniform(r[i])<prob,Val(4));end
  src=let pos=linear,t=threads,total=n,arr=a;ntuple(i->(@inbounds pos+t*UInt32(i-1)<total ? arr[pos+t*UInt32(i-1)+1] : 0f0),Val(4));end
  @inbounds for lane in 0:3
   li=linear+threads*UInt32(lane)
   if li<n;b[li+1]=src[lane+1]*Float32(keep[lane+1])*scale;mask[li+1]=UInt8(keep[lane+1]);end
  end
  JACC.sync_workgroup();linear+=stride
 end
 return
end
function dropout_vector!(global_idx,::Val{V},a,b,mask,n,p,seed,offset,total_threads) where V
 idx=UInt32(global_idx-1);block0=UInt32(div(global_idx-1,256));threads=total_threads;stride=threads*UInt32(V);linear=idx*UInt32(V);first=block0*UInt32(256*V);loops=first<n ? (n-UInt32(1)-first)÷stride+UInt32(1) : UInt32(0);scale=1f0/p;counter,cache=initial_rng(seed,UInt64(idx),offset)
 for iteration in UInt32(1):loops
  if linear<n
   counter,cache,raw=rng_next(counter,cache,seed,offset);keep=let r=raw,prob=p;ntuple(i->rng_uniform(r[i])<prob,Val(V));end;index=linear÷UInt32(V)+UInt32(1);src=@inbounds a[index]
   result=let values=src,bits=keep,s=scale;ntuple(i->VecElement(values[i].value*Float32(bits[i])*s),Val(V));end
   bits=let flags=keep;ntuple(i->VecElement(UInt8(flags[i])),Val(V));end
   @inbounds b[index]=result;@inbounds mask[index]=bits
  end
  JACC.sync_workgroup();linear+=stride
 end
 return
end
function rng_probe_kernel!(global_idx,out,subseqs,seed,offset)
 i=Int32(global_idx)
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
 blocks=vec==1 ? 512 : vec==2 ? 256 : 128;spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 if vec==1
  JACC.parallel_for(spec,blocks*256,dropout_scalar!,a,b,mask,n,p,seed,offset,UInt32(blocks*256))
 else
  JACC.parallel_for(spec,blocks*256,dropout_vector!,Val(vec),a,b,mask,n,p,seed,offset,UInt32(blocks*256))
 end
end
function probe(out,seq,seed,offset)
 spec=JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false)
 JACC.parallel_for(spec,32,rng_probe_kernel!,out,seq,seed,offset)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch,probe)
