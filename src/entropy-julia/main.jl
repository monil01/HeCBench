include(joinpath(@__DIR__,"entropy-julia-cuda.jl","main.jl"))
study_main(entropy_launch!)
