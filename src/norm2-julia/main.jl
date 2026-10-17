# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"norm2-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION original_firstTwoLengths_524288_1048576_host_result_eachcall_v1")
exit(study_main(study_prepare,ARGS))
