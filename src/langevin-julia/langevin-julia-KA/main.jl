using CUDA,KernelAbstractions,Printf

@kernel function portable0!(a,o,n)
 t=@index(Global,Linear)
 if t<=n
  x=a[t];o[t]=cosh(x)/sinh(x)-1f0/x
 end
end
@kernel function portable1!(a,o,n)
 t=@index(Global,Linear)
 if t<=n
  x=a[t];o[t]=1f0/tanh(x)-1f0/x
 end
end
@kernel function portable2!(a,o,n)
 t=@index(Global,Linear)
 if t<=n
  x=a[t];s=x*x;r=7.70960469f-8
  r=fma(r,s,-1.65101926f-6);r=fma(r,s,2.03457112f-5)
  r=fma(r,s,-2.10521728f-4);r=fma(r,s,2.11580913f-3)
  r=fma(r,s,-2.22220998f-2);r=fma(r,s,8.33333284f-2)
  o[t]=fma(r,x,.25f0*x)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(a,outs)
 kernels=(portable0!(CUDABackend(),256),portable1!(CUDABackend(),256),portable2!(CUDABackend(),256))
 mode->kernels[mode](a,outs[mode],N;ndrange=N)
end
langevin_study(factory)
