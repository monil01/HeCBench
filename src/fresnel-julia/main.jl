# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"fresnel-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION reference_repaired_weightedIntegralSinTSquared_grid0to8_step1over1024_v1")
exit(study_main(study_prepare,ARGS))
