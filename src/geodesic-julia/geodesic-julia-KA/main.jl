using CUDA
using KernelAbstractions
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_math.jl"))
const F4=NTuple{4,VecElement{Float32}}
@kernel function distance_kernel!(input,out,n::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n
  @inbounds v=input[i]
  @inbounds out[i]=distance_one(v[1].value,v[2].value,v[3].value,v[4].value)
 end
end
launch(input,out,n)=distance_kernel!(BACKEND,256)(input,out,n;ndrange=cld(Int(n),256)*256)
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
