using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(p,g,gn,w,m,q,out,::Type{T}) where T
 x=@index(Global,Linear)
 if x<=HH
  tmp,sv,idx=study_value(T,p,g,gn,w,m,q,x)
  @inbounds for c in Int32(0):Int32(WW-1)
   off=c*Int32(HH)+x;z=Float32(study_exp(p[off])*sv);out[off]=T(c==idx ? Float32(tmp)-z : -z)
  end
 end
end
run!(args...)=kernel!(get_backend(args[7]),64)(args...;ndrange=HH)
study(run!)
