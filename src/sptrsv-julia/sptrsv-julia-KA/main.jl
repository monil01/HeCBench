using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function solve!(rp,ci,v,flags,m,b,x,warpmap,len)
 sums=@localmem Float64 (256,)
 localid=Int32(@index(Local,Linear))-Int32(1)
 globalid=Int32(@index(Global,Linear))-Int32(1)
 solve_body!(rp,ci,v,flags,m,b,x,warpmap,len,localid,globalid,sums)
end
function launch(rp,ci,v,flags,m,b,x,warpmap,len)
 solve!(BACKEND,256)(rp,ci,v,flags,m,b,x,warpmap,len;ndrange=cld(Int(len)-1,8)*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(v->begin a=KernelAbstractions.allocate(BACKEND,eltype(v),length(v));copyto!(a,v);a end,launch)
