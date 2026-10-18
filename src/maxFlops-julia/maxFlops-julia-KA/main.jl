using CUDA,KernelAbstractions
include("../arithmetic_device.jl")
@kernel unsafe_indices=true function arithmetic_kernel!(data,nIters,v1,v2,op,L)
 arithmetic_site!(data,Int(@index(Global,Linear)),nIters,v1,v2,op,L)
end
const AK=arithmetic_kernel!(CUDABackend(),256)
launch(c)=(AK(c.data,c.R,c.v1,c.v2,Val(c.op),Val(c.L);ndrange=c.N);nothing)
include("../study_common.jl")
arithmetic_study(launch)
