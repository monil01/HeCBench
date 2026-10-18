include(joinpath(@__DIR__,"resnet-kernels-julia-cuda.jl","main.jl"))
study_main("cuda_jl",launch!)
