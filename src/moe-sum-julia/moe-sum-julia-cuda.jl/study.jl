using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function moe_sum_ref!")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,output,vec)
    if vec
        return ()->(@cuda threads=1024 blocks=512 moe_sum_vec4_kernel!(output,input,Int32(1024),Int32(3)))
    else
        return ()->(@cuda threads=1024 blocks=512 moe_sum_kernel!(output,input,Int32(1024),Int32(3)))
    end
end
moe_study(factory)
