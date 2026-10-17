using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"function main()")),joinpath(@__DIR__,"main.jl"))
launch_julia!(o,a,s,n)=(@cuda threads=Int(n) blocks=1 binary_scan_kernel!(o,a,n);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"));bscan_study(launch_julia!)
