using CUDA,KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function su3!(a,b,c,n)
 id=Int32(@index(Global,Linear))-Int32(1)
 su3_body!(a,b,c,n,id)
end
function launch(a,b,c,n,threads)
 su3!(BACKEND,threads)(a,b,c,n;ndrange=Int(n)*threads)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(v->begin a=KernelAbstractions.allocate(BACKEND,UInt32,length(v));copyto!(a,v);a end,launch)
