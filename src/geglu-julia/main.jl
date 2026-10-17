# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"geglu-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION n128_allThreeWidths_1280_2560_5120_FP32_erf_vectorLaneMap_v1")
exit(study_main(study_prepare,ARGS))
