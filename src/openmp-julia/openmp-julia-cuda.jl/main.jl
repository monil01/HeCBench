using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(a,b,repeat)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 for j in Int32(0):repeat-Int32(1)
  @inbounds a[i]+=mod(j,b)
 end
 return
end
run!(a,b,r)=(@cuda threads=256 blocks=length(a)÷256 kernel!(a,b,r))
study(run!)
