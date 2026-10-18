using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
const TEMPLATE=JACC.zeros(Float64,256)
function solve!(i,rp,ci,v,flags,m,b,x,warpmap,len,template)
 sums=JACC.shared(template)
 JACC.sync_workgroup() # Uniform shared-template initialization before any solver return.
 localid=Int32((i-1)%256);globalid=Int32(i)-Int32(1)
 solve_body!(rp,ci,v,flags,m,b,x,warpmap,len,localid,globalid,sums)
 return nothing
end
function launch(rp,ci,v,flags,m,b,x,warpmap,len)
 blocks=cld(Int(len)-1,8);spec=JACC.launch_spec(;threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*256,solve!,rp,ci,v,flags,m,b,x,warpmap,len,TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
