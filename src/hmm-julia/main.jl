include(joinpath(@__DIR__,"hmm-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Source-faithful positive max-plus score DP, notnormalizedHMMlikelihood; matchedD2Dcopies")
exit(study_main(study_prepare))
