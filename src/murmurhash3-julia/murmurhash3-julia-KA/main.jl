using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function murmur_ka!(keys,offsets,lengths,out,num_keys::UInt32)
 i0=UInt32(@index(Global,Linear))-UInt32(1)
 if i0<num_keys
  idx=i0+UInt32(1);@inbounds base=offsets[idx];@inbounds len=lengths[idx];h1,h2=murmur_hash(keys,base,len,i0)
  @inbounds out[i0*UInt32(2)+UInt32(1)]=h1
  @inbounds out[i0*UInt32(2)+UInt32(2)]=h2
 end
end
function factory(keys,offsets,lengths,out,n)
 k=murmur_ka!(CUDABackend(),256)
 ()->k(keys,offsets,lengths,out,UInt32(n);ndrange=n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hash_study(factory)
