include(joinpath(@__DIR__,"divergence-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION NP4 manufactured polynomial full host workflow")
exit(study_main(study_prepare))
