using CUDA
include(joinpath(@__DIR__,"..","study_math.jl"))
const F4=NTuple{4,VecElement{Float32}}
function distance_kernel!(input,out,n::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n
  @inbounds v=input[i]
  @inbounds out[i]=distance_one(v[1].value,v[2].value,v[3].value,v[4].value)
 end
 return
end
launch(input,out,n)=(@cuda threads=256 blocks=cld(Int(n),256) distance_kernel!(input,out,n))
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
