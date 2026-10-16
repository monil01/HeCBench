using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
source=source[findfirst("function bincount_global_kernel!",source).start:findfirst("function reference!",source).start-1]
include(joinpath(@__DIR__,"..","study_common.jl"));include_string(Main,source,joinpath(@__DIR__,"main.jl"))
function factory(output,input,shared)
    if shared
        return ()->(@cuda threads=256 blocks=1024 shmem=3072 bincount_shared_kernel!(output,input,Int32(768),0f0,1f0,Int32(262144)))
    else
        return ()->(@cuda threads=256 blocks=1024 bincount_global_kernel!(output,input,Int32(768),0f0,1f0,Int32(262144)))
    end
end
bincount_study(factory)
