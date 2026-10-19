using CUDA, KernelAbstractions
include("kernel.jl")
include(joinpath(@__DIR__,"..","study_host.jl"))
function launch(frame,common,buffers)
    hw(KernelAbstractions.get_backend(buffers.frame),256)(frame,common,buffers;ndrange=51*256)
    nothing
end
study_main(launch,CuArray)
