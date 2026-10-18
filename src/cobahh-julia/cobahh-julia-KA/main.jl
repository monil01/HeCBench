using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
const BACKEND=CUDA.CUDABackend()
@kernel function neuron_kernel!(args...)
 i=Int32(@index(Global,Linear))
 point!(i,args...)
end
function launch(h,m,n,ge,v,gi,last,nr,N,dt,t,P)
 neuron_kernel!(BACKEND,256)(h,m,n,ge,v,gi,last,nr,N,dt,t,P;ndrange=cld(Int(N),256)*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
