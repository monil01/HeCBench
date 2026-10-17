using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nmain\(\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
aidw_study((dx,dy,dz,dnum,ix,iy,iz,inum,area,avg)->()->(@cuda threads=256 blocks=cld(Int(inum),256) aidw_kernel!(dx,dy,dz,dnum,ix,iy,iz,inum,area,avg)))
