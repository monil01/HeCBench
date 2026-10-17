using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
launch_julia!(a,ci,blocks)=(@cuda threads=256 blocks=blocks benchmark_kernel!(a,ci);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"));mixbench_study(launch_julia!)
