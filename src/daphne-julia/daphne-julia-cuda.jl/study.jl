using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"function main()")),joinpath(@__DIR__,"main.jl"))
launch_julia!(cp,x,y,z,v,n)=(@cuda threads=256 blocks=cld(n,256) project_kernel!(cp,x,y,z,v,n);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"));daphne_study(launch_julia!)
