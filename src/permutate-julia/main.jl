# Unmeasured compatibility launcher; Python central verifier owns GPU6 lock.
run(`python3 $(joinpath(@__DIR__,"run_study.py")) --models cuda_jl`)
