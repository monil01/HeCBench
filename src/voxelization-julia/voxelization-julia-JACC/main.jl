using CUDA
import JACC
JACC.@init_backend
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function build!(idx,points,n,hash,real)
 build_hash_body!(Int32(idx-1),points,n,hash,real)
 nothing
end
@inline function scatter!(idx,points,n,hash,count,temp,index)
 voxelization_body!(Int32(idx-1),points,n,hash,count,temp,index)
 nothing
end
@inline function feature!(idx,temp,count,real,features)
 feature_extraction_body!(Int32(idx-1),temp,count,real,features)
 nothing
end
function build_scatter(c)
 range=cld(c.n,256)*256
 spec=JACC.launch_spec(;threads=256,blocks=cld(c.n,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,range,build!,c.points,Int32(c.n),c.hash,c.real)
 JACC.parallel_for(spec,range,scatter!,c.points,Int32(c.n),c.hash,c.count,c.temp,c.indices)
end
function extract(c,real)
 spec=JACC.launch_spec(;threads=256,blocks=cld(real,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,real,feature!,c.temp,c.count,UInt32(real),c.features)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->JACC.zeros(UInt32,n),build_scatter,extract,"jacc")
