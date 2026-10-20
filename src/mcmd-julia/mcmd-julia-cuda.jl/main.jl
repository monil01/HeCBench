using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","mcmd_device.jl"))
function force_entry!(a,n,cut,b,rb,alpha,km,ks)
    i=Int32(threadIdx().x)+Int32(blockIdx().x-1)*Int32(blockDim().x)
    mcmd_force_workitem!(i,a,n,cut,b,rb,alpha,km,ks)
    nothing
end
function mcmd_launch(a,n,grid,block,cut,b,rb,alpha,km,ks)
    CUDA.@cuda threads=block blocks=grid stream=CUDA.legacy_stream() force_entry!(a,n,cut,b,rb,alpha,km,ks)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mcmd_study()
