using CUDA,KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function rush_kernel!(s,t,dt,p,n)
 i=Int32(@index(Global,Linear))-Int32(1)
 rush_body!(s,t,dt,p,n,i)
end
function launch(c,t)
 rush_kernel!(BACKEND,256)(c.ds,t,c.dt,c.dp,c.n;ndrange=cld(Int(c.n),256)*256)
end
function alloc(v)
 a=KernelAbstractions.allocate(BACKEND,Float64,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rl_study(alloc,launch)
