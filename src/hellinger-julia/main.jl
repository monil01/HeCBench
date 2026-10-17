# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"hellinger-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION normalized_probability_M32N64P48_uniformAndNonuniformA_FP32_v1")
exit(study_main(study_prepare,ARGS))
