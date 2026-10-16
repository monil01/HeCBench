using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function ref_nhwc!")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,sum,sumsq,nchw)
    if nchw
        return ()->(@cuda threads=256 blocks=128 channel_sum_nchw_kernel!(Int32(4),Int32(128),Int32(1024),input,sum,sumsq))
    else
        return ()->(@cuda threads=256 blocks=128 channel_sum_nhwc_kernel!(Int32(4),Int32(128),Int32(1024),input,sum,sumsq))
    end
end
channel_study(factory)
