using CUDA, KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function build!(points,n,hash,real)
 idx=@index(Global,Linear)
 build_hash_body!(Int32(idx-1),points,n,hash,real)
end
@kernel function scatter!(points,n,hash,count,temp,index)
 idx=@index(Global,Linear)
 voxelization_body!(Int32(idx-1),points,n,hash,count,temp,index)
end
@kernel function feature!(temp,count,real,features)
 idx=@index(Global,Linear)
 feature_extraction_body!(Int32(idx-1),temp,count,real,features)
end
function build_scatter(c)
 range=cld(c.n,256)*256
 build!(CUDA.CUDABackend(),256)(c.points,Int32(c.n),c.hash,c.real;ndrange=range)
 scatter!(CUDA.CUDABackend(),256)(c.points,Int32(c.n),c.hash,c.count,c.temp,c.indices;ndrange=range)
end
function extract(c,real)
 feature!(CUDA.CUDABackend(),256)(c.temp,c.count,UInt32(real),c.features;ndrange=real)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->KernelAbstractions.allocate(CUDA.CUDABackend(),UInt32,(n,)),build_scatter,extract,"ka")
