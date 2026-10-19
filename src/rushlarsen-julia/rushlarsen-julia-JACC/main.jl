using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function rush_kernel!(i,s,t,dt,p,n)
 rush_body!(s,t,dt,p,n,Int32(i)-Int32(1));return
end
function launch(c,t)
 spec=JACC.launch_spec(threads=256,blocks=cld(Int(c.n),256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,Int(c.n),rush_kernel!,c.ds,t,c.dt,c.dp,c.n)
end
function alloc(v)
 a=JACC.zeros(Float64,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rl_study(alloc,launch)
