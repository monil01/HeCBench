include("overlay-julia-cuda.jl/main.jl")
if abspath(PROGRAM_FILE)==@__FILE__;main();end
