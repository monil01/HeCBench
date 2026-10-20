# Unmeasured compatibility entrypoint. The Python verifier owns the GPU lock.
run(`python3 $(joinpath(@__DIR__,"run_study.py")) --models cuda_jl`)
