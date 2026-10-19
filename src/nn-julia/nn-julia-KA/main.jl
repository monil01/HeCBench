using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function nn_kernel!(n::Int32,lat::Float32,lng::Float32,locations,distances)
 gid=Int32(@index(Global,Linear))-Int32(1)
 if gid<n
  @inbounds p=locations[gid+Int32(1)]
  dx=lat-p.lat;dy=lng-p.lng
  @inbounds distances[gid+Int32(1)]=sqrt(dx*dx+dy*dy)
 end
end
@kernel function echo_parameters!(out,n::Int32,lat::Float32,lng::Float32)
 @inbounds begin;out[1]=lat;out[2]=lng;out[3]=Float32(n);end
end
const NN_KERNEL=nn_kernel!(CUDABackend(),64)
const ECHO_KERNEL=echo_parameters!(CUDABackend(),1)
nn_study((p,o,n,lat,lng)->NN_KERNEL(n,lat,lng,p,o;ndrange=Int(n)),(out,n,lat,lng)->ECHO_KERNEL(out,n,lat,lng;ndrange=1))
