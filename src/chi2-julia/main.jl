# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"chi2-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION reference_repaired_Pearson_margins_rows12cols257_case3control9_twoFixtures_v1")
exit(study_main(study_prepare,ARGS))
