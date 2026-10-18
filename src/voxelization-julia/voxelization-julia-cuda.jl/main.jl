using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function build!(points,n,hash,real)
 idx=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 build_hash_body!(idx,points,n,hash,real)
 return
end
function scatter!(points,n,hash,count,temp,index)
 idx=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 voxelization_body!(idx,points,n,hash,count,temp,index)
 return
end
function feature!(temp,count,real,features)
 idx=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 feature_extraction_body!(idx,temp,count,real,features)
 return
end
function build_scatter(c)
 @cuda threads=256 blocks=cld(c.n,256) build!(c.points,Int32(c.n),c.hash,c.real)
 @cuda threads=256 blocks=cld(c.n,256) scatter!(c.points,Int32(c.n),c.hash,c.count,c.temp,c.indices)
end
function extract(c,real)
 @cuda threads=256 blocks=cld(real,256) feature!(c.temp,c.count,UInt32(real),c.features)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(n->CUDA.CuArray{UInt32}(undef,n),build_scatter,extract,"cuda_jl")
