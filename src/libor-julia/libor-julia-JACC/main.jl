using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function libor_jacc!(i,::Val{GREEK},v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb) where GREEK
 tid=Int32(i)-Int32(1)
 if GREEK
  greek_body!(tid,Int32(256),v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 else
  value_body!(tid,Int32(256),v,lambda,maturities,rates,delta,L,B,S)
 end
 nothing
end
const SPEC=JACC.launch_spec(threads=64,blocks=4,shmem_size=0,sync=false)
function run!(mode,v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 JACC.parallel_for(SPEC,256,libor_jacc!,mode,v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 nothing
end
study(run!)
