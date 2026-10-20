# Compatibility entrypoint; use the pinned CUDA.jl project and built study_host.so.
include(joinpath(@__DIR__,"hpl-julia-cuda.jl","main.jl"))
