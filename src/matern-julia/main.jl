include("matern-julia-cuda.jl/main.jl")
if abspath(PROGRAM_FILE)==@__FILE__; exit(main(ARGS)); end
