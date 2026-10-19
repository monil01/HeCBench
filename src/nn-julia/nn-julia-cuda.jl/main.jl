using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function nn_kernel!(n::Int32,lat::Float32,lng::Float32,locations,distances)
 gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 if gid<n
  @inbounds p=locations[gid+Int32(1)]
  dx=lat-p.lat;dy=lng-p.lng
  @inbounds distances[gid+Int32(1)]=sqrt(dx*dx+dy*dy)
 end
 return
end
function echo_parameters!(out,n::Int32,lat::Float32,lng::Float32)
 @inbounds begin;out[1]=lat;out[2]=lng;out[3]=Float32(n);end
 return
end
nn_study((p,o,n,lat,lng)->(@cuda threads=64 blocks=cld(n,64) nn_kernel!(n,lat,lng,p,o)),(out,n,lat,lng)->(@cuda threads=1 blocks=1 echo_parameters!(out,n,lat,lng)))
