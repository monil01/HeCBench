using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_math.jl"))
function neuron_kernel!(i,args...)
 point!(Int32(i),args...)
 return
end
function launch(h,m,n,ge,v,gi,last,nr,N,dt,t,P)
 blocks=cld(Int(N),256)
 spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,neuron_kernel!,h,m,n,ge,v,gi,last,nr,N,dt,t,P)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
