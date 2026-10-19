using KernelAbstractions
include(joinpath(@__DIR__,"..","photon_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
# Explicit CUDA backend extensions preserve the source's warp-synchronous RNG,
# rejection sampling, static shared storage and Float32 detector atomics.
@kernel function simulation_ka(c,initialize,limit)
    lane=Int32(@index(Local,Linear))-Int32(1)
    @inbounds egs_transport(c,lane,initialize,limit)
    nothing
end
@kernel function accumulation_ka(c)
    block=Int32(@index(Group,Linear))-Int32(1)
    lane=Int32(@index(Local,Linear))-Int32(1)
    @inbounds egs_accumulate(c,block,lane)
    nothing
end
const EGS_KA_BACKEND=CUDA.CUDABackend()
launch_simulation(c,initialize,limit)=simulation_ka(EGS_KA_BACKEND,32)(c,initialize,limit;ndrange=32)
launch_accumulation(c)=accumulation_ka(EGS_KA_BACKEND,1024)(c;ndrange=8192)
egs_study()
