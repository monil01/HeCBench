# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"blas-gemm-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION binary_srand2_M64K32N48_FP16FP32FP64_alpha2_betaHalf_reset10_v1")
exit(study_main(study_prepare,ARGS))
