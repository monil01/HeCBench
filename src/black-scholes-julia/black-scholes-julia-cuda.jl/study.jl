using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nmain\(\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
bs_study((ty,S,K,q,r,T,v,out,n)->()->(@cuda threads=256 blocks=cld(Int(n),256) bs_kernel!(ty,S,K,q,r,T,v,out,n)))
