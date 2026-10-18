using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function murmur_jacc!(index,keys,offsets,lengths,out,num_keys::UInt32)
 i0=UInt32(index)-UInt32(1)
 if i0<num_keys
  idx=i0+UInt32(1);@inbounds base=offsets[idx];@inbounds len=lengths[idx];h1,h2=murmur_hash(keys,base,len,i0)
  @inbounds out[i0*UInt32(2)+UInt32(1)]=h1
  @inbounds out[i0*UInt32(2)+UInt32(2)]=h2
 end
 return
end
function factory(keys,offsets,lengths,out,n)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,n,murmur_jacc!,keys,offsets,lengths,out,UInt32(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hash_study(factory)
