include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run!(::Val{GREEK},v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb) where GREEK
 if GREEK
  @cuda threads=64 blocks=4 pathcalc_portfolio_kernel!(v,g,lambda,maturities,rates,delta,L,L2,B,S,Bb,Sb)
 else
  @cuda threads=64 blocks=4 pathcalc_portfolio_kernel2!(v,lambda,maturities,rates,delta,L,B,S)
 end
 nothing
end
study(run!)
