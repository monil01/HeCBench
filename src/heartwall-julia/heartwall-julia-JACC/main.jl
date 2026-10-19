using CUDA
import JACC
JACC.@init_backend
include("kernel.jl")
include(joinpath(@__DIR__,"..","study_host.jl"))
function launch(frame,common,buffers)
    JACC.parallel_for(JACC.launch_spec(threads=256,blocks=51,shmem_size=0,sync=false),51*256,hw,frame,common,buffers)
    nothing
end
study_main(launch,JACC.array)
