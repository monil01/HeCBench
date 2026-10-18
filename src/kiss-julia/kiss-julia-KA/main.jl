using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function rng_kernel!(out,n::UInt64,seed::UInt32)
 tid=reinterpret(UInt32,Int32(@index(Global,Linear))-Int32(1))
 kiss_body!(out,n,seed,tid,UInt64(1048576))
end
function launch(out,n,seed)
 rng_kernel!(BACKEND,256)(out,n,seed;ndrange=1048576)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
