include(joinpath(@__DIR__,"convolution1D-julia-cuda.jl","main.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;exit(main());end
