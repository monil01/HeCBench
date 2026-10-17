# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"cooling-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION positive_temperature_log1e3to1e8_n512_density00899_heat0and1_v1")
exit(study_main(study_prepare,ARGS))
