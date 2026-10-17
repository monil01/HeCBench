# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"blas-dot-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION constant_sqrt65504_n256_FP64FP32FP16BF16_dotEx_repeat100_v1")
exit(study_main(study_prepare,ARGS))
