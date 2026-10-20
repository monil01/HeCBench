using CUDA,Libdl,SHA,KernelAbstractions
include(joinpath(@__DIR__,"..","mcmd_device.jl"))
@kernel function force_entry!(a,n,cut,b,rb,alpha,km,ks)
    i=Int32(@index(Global,Linear))
    mcmd_force_workitem!(i,a,n,cut,b,rb,alpha,km,ks)
end
const MCMD_KA_BACKEND=CUDA.CUDABackend(always_inline=false)
function mcmd_launch(a,n,grid,block,cut,b,rb,alpha,km,ks)
    kernel=force_entry!(MCMD_KA_BACKEND,block)
    kernel(a,n,cut,b,rb,alpha,km,ks;ndrange=grid*block)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mcmd_study()
