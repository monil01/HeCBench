using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
const F4=NTuple{4,VecElement{Float32}}
function distance_kernel!(index,input,out,n::Int32)
 i=Int32(index)
 if i<=n
  @inbounds v=input[i]
  @inbounds out[i]=distance_one(v[1].value,v[2].value,v[3].value,v[4].value)
 end
 return
end
function launch(input,out,n)
 N=cld(Int(n),256)*256;spec=JACC.launch_spec(threads=256,blocks=cld(Int(n),256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,N,distance_kernel!,input,out,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
