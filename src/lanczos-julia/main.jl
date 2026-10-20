# Scoped compatibility entrypoint: use the pinned CUDA.jl project and built study_host.so.
include(joinpath(@__DIR__,"lanczos-julia-cuda.jl","main.jl"))
