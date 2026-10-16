using CUDA
ENV["HECBENCH_STUDY_INCLUDE_ONLY"]="1"
include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,output,layout)
    if layout=="aos"
        return ()->(@cuda threads=256 blocks=16 aos_kernel!(input,output))
    end
    return ()->(@cuda threads=256 blocks=16 soa_kernel!(input,output))
end
layout_study(factory)
