using CUDA,Libdl,SHA,JACC
include(joinpath(@__DIR__,"..","mcmd_device.jl"))
@inline function force_entry!(index,a,n,cut,b,rb,alpha,km,ks)
    mcmd_force_workitem!(Int32(index),a,n,cut,b,rb,alpha,km,ks)
    nothing
end
const MCMD_SPECS=Dict{Tuple{Int,Int},Any}()
function mcmd_launch(a,n,grid,block,cut,b,rb,alpha,km,ks)
    spec=get!(MCMD_SPECS,(block,grid)) do
        JACC.launch_spec(;threads=(block,),blocks=(grid,),shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
    JACC.parallel_for(spec,grid*block,force_entry!,a,n,cut,b,rb,alpha,km,ks)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mcmd_study()
