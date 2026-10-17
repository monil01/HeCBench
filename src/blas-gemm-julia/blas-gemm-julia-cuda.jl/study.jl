include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
println("STUDY_LIBRARY Julia_CUBLAS ",CUDA.CUBLAS.version())
exit(study_main(study_prepare))
