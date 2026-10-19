using JACC
include(joinpath(@__DIR__,"..","photon_device.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
# CUDA-specific warp collectives, static shared memory and atomics are explicit
# backend extensions; all photon computations are authored Julia device code.
@inline function simulation_jacc(i,c,initialize,limit)
    lane=Int32(i)-Int32(1)
    @inbounds egs_transport(c,lane,initialize,limit)
    nothing
end
@inline function accumulation_jacc(i,c)
    index=Int32(i)-Int32(1)
    @inbounds egs_accumulate(c,index÷Int32(1024),index%Int32(1024))
    nothing
end
# Complete explicit grids prevent inferred-grid reuse; dynamic shared memory is
# zero, while the typed static shared arrays in egs_transport remain allocated.
const EGS_SIM_SPEC=JACC.launch_spec(;threads=32,blocks=1,shmem_size=0,sync=false)
const EGS_SUM_SPEC=JACC.launch_spec(;threads=1024,blocks=8,shmem_size=0,sync=false)
launch_simulation(c,initialize,limit)=JACC.parallel_for(EGS_SIM_SPEC,32,simulation_jacc,c,initialize,limit)
launch_accumulation(c)=JACC.parallel_for(EGS_SUM_SPEC,8192,accumulation_jacc,c)
egs_study()
