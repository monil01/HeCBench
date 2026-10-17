include(joinpath(@__DIR__,"hadamard-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Named full Hadamard study: all five dimensions, FP32/FP16/BF16, batch2")
exit(study_main(study_prepare))
