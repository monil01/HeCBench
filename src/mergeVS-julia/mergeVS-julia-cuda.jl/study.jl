include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(p)
 f,ins,outs,causal=p;B,H,R,M,BN,V,S=f[1:7]
 @cuda threads=64 blocks=(H,B,cld(R,64)) metadata_study!(ins...,outs...,H,R,M,BN,V,S,causal)
end
study_main(launch!)
