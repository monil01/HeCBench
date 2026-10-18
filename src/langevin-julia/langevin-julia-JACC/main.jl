
using CUDA,JACC,Printf
JACC.@init_backend
function portable0!(t,a,o,n)
 if t<=n
  x=a[t];o[t]=cosh(x)/sinh(x)-1f0/x
 end
 nothing
end
function portable1!(t,a,o,n)
 if t<=n
  x=a[t];o[t]=1f0/tanh(x)-1f0/x
 end
 nothing
end
function portable2!(t,a,o,n)
 if t<=n
  x=a[t];s=x*x;r=7.70960469f-8
  r=fma(r,s,-1.65101926f-6);r=fma(r,s,2.03457112f-5)
  r=fma(r,s,-2.10521728f-4);r=fma(r,s,2.11580913f-3)
  r=fma(r,s,-2.22220998f-2);r=fma(r,s,8.33333284f-2)
  o[t]=fma(r,x,.25f0*x)
 end
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(a,outs)
 spec=JACC.launch_spec(;threads=256,blocks=16,shmem_size=0,sync=false)
 kernels=(portable0!,portable1!,portable2!)
 mode->JACC.parallel_for(spec,N,kernels[mode],a,outs[mode],N)
end
langevin_study(factory)
