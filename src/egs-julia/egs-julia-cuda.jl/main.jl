include(joinpath(@__DIR__,"..","photon_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function simulation_kernel(c,initialize,limit)
    lane=Int32(threadIdx().x)-Int32(1)
    @inbounds egs_transport(c,lane,initialize,limit)
    return
end
function accumulation_kernel(c)
    block=Int32(blockIdx().x)-Int32(1); lane=Int32(threadIdx().x)-Int32(1)
    @inbounds egs_accumulate(c,block,lane)
    return
end
launch_simulation(c,initialize,limit)=(@cuda threads=32 blocks=1 simulation_kernel(c,initialize,limit))
launch_accumulation(c)=(@cuda threads=1024 blocks=8 accumulation_kernel(c))
egs_study()
