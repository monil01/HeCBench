using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function nn_kernel!(index,n::Int32,lat::Float32,lng::Float32,locations,distances)
 gid=Int32(index)-Int32(1)
 if gid<n
  @inbounds p=locations[gid+Int32(1)]
  dx=lat-p.lat;dy=lng-p.lng
  @inbounds distances[gid+Int32(1)]=sqrt(dx*dx+dy*dy)
 end
 return
end
function echo_parameters!(index,out,n::Int32,lat::Float32,lng::Float32)
 @inbounds begin;out[1]=lat;out[2]=lng;out[3]=Float32(n);end
 return
end
const NN_SPECS=Dict(c.n=>JACC.launch_spec(;threads=64,blocks=cld(Int(c.n),64),shmem_size=0,sync=false) for c in CASES)
const ECHO_SPEC=JACC.launch_spec(;threads=1,blocks=1,shmem_size=0,sync=false)
nn_study((p,o,n,lat,lng)->JACC.parallel_for(NN_SPECS[n],Int(n),nn_kernel!,n,lat,lng,p,o),(out,n,lat,lng)->JACC.parallel_for(ECHO_SPEC,1,echo_parameters!,out,n,lat,lng))
