using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(p,g,gn,w,m,q,out,::Type{T}) where T
 x=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if x<=HH
  tmp,sv,idx=study_value(T,p,g,gn,w,m,q,x)
  @inbounds for c in Int32(0):Int32(WW-1)
   off=c*Int32(HH)+x;z=Float32(study_exp(p[off])*sv);out[off]=T(c==idx ? Float32(tmp)-z : -z)
  end
 end
 return
end
run!(args...)=(@cuda threads=64 blocks=cld(HH,64) kernel!(args...))
study(run!)
