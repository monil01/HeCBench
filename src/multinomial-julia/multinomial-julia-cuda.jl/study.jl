using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
launch_julia!(args...)=(@cuda threads=256 blocks=512 sample_multinomial_once_kernel!(args...);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"));multinomial_study(launch_julia!)
