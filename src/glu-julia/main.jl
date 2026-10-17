# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"glu-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION tensor32cubed_allseven_nativeAxisLoop_Float32_boundedFixture_v1")
exit(study_main(study_prepare,ARGS))
