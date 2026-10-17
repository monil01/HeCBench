# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"blas-groupgemm-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION MoE_FourExperts_K32N48_tokens32_nativeRouting_FP16_groupedEx129_v1")
exit(study_main(study_prepare,ARGS))
