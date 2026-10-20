# Unmeasured compatibility entrypoint; central verifier owns GPU lock.
run(`python3 $(joinpath(@__DIR__,"run_study.py")) --models cuda_jl`)
