# Convenience CUDA.jl study launcher; the central verifier owns GPU6 locking.
# Input is an already prepared/independently gated fixture directory, not steps.
fixtures=isempty(ARGS) ? "/tmp/hecbench-stencil-ports-rbc-fixtures" : abspath(ARGS[1])
config=tempname()*".json"
run(`python3 $(joinpath(@__DIR__,"make_config.py")) --fixtures $fixtures --models cuda_jl --output $config`)
run(`python3 $(joinpath(@__DIR__,"run_study.py")) --config $config`)
