using CUDA
include("kernel.jl")
include(joinpath(@__DIR__,"..","study_host.jl"))
launch(frame,common,buffers)=(@cuda threads=256 blocks=51 hw(frame,common,buffers);nothing)
study_main(launch,CuArray)
