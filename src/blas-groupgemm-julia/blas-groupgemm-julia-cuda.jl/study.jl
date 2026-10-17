include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
println("STUDY_LIBRARY explicit_CUBLAS ",STUDY_CUBLAS_LIBRARY)
exit(study_main(study_prepare))
