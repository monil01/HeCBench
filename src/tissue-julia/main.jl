include(joinpath(@__DIR__,"tissue-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION dim8 two_species full_operator native_step1")
exit(study_main(study_prepare))
