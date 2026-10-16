using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
# Preserve CUDA.jl mapping: one work item/block for each pair, sequential attributes.
function apd_jacc!(gx,gy,distance,data,instances::Int32,attributes::Int32)
    if gx<=instances&&gy<=instances
        count=Int32(0)
        for i in Int32(1):attributes
            count += data[i+attributes*(gx-1)]!=data[i+attributes*(gy-1)]
        end
        distance[instances*(gx-1)+gy]=count
    end
end
function factory(output,input,instances,attributes)
    spec=JACC.launch_spec(;threads=(1,1),sync=false)
    return ()->JACC.parallel_for(spec,(instances,instances),apd_jacc!,output,input,Int32(instances),Int32(attributes))
end
apd_study(factory)
