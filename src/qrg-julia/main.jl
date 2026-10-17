include(joinpath(@__DIR__,"qrg-julia-cuda.jl","main.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;main();end
