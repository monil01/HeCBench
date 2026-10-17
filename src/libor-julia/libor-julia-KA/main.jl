using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function libor_ka!(::Val{GREEK},v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb) where GREEK
 tid=Int32(@index(Global,Linear))-Int32(1)
 if GREEK
  greek_body!(tid,Int32(256),v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 else
  value_body!(tid,Int32(256),v,lambda,maturities,rates,delta,L,B,S)
 end
end
const BACKEND=CUDABackend()
function run!(mode,v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 libor_ka!(BACKEND,64)(mode,v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb;ndrange=256)
 nothing
end
study(run!)
