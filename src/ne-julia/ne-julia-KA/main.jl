using CUDA,KernelAbstractions
include("../normal_device.jl")
@kernel unsafe_indices=true function normal_kernel!(points,out,W,H)
 normal_site!(points,out,Int(@index(Global,Linear)),W,H)
end
const NK=normal_kernel!(CUDABackend(),256)
launch(c)=(NK(c.p,c.out,c.W,c.H;ndrange=cld(c.N,256)*256);nothing)
include("../study_common.jl")
normal_study(launch)
