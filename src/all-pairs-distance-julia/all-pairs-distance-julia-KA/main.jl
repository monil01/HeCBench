using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
# Preserve CUDA.jl mapping: one work item/block for each pair, sequential attributes.
@kernel function apd_ka!(distance,data,instances::Int32,attributes::Int32)
    gx,gy=@index(Global,NTuple)
    if gx<=instances&&gy<=instances
        count=Int32(0)
        for i in Int32(1):attributes
            count += data[i+attributes*(gx-1)]!=data[i+attributes*(gy-1)]
        end
        distance[instances*(gx-1)+gy]=count
    end
end
function factory(output,input,instances,attributes)
    kernel=apd_ka!(get_backend(input),(1,1))
    return ()->kernel(output,input,Int32(instances),Int32(attributes);ndrange=(instances,instances))
end
apd_study(factory)
