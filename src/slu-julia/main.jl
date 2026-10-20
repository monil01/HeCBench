# Unmeasured convenience forwarding entrypoint; frozen commands use the actual child.
length(ARGS)==3 || error("Usage: julia --project=src/_julia_ka_env src/slu-julia/main.jl STUDY_LIBRARY FIXTURE_ROOT PRIVATE_CAPTURE_ROOT; four-way study: python3 src/slu-julia/run_four_models.py --fixtures FIXTURE_ROOT")
include(joinpath(@__DIR__,"slu-julia-cuda.jl","main.jl"))
