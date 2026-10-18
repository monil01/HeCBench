using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function rng_kernel!(xx,out,n::UInt64,seed::UInt32)
 tid=reinterpret(UInt32,Int32(xx)-Int32(1))
 kiss_body!(out,n,seed,tid,UInt64(1048576))
 return nothing
end
function launch(out,n,seed)
 spec=JACC.launch_spec(;threads=256,blocks=4096,shmem_size=0,sync=false)
 JACC.parallel_for(spec,1048576,rng_kernel!,out,n,seed)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
