using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
# Define only original device kernels; common study contains the original CPU routine.
include_string(Main,first(split(source,"function log_probs_cpu")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(out,total,input,ids,lengths)
    return ()->begin
        @cuda threads=256 blocks=(31,64) log_probs_kernel!(out,input,ids,lengths,Int32(32),Int32(64),Int32(256),Int32(256))
        @cuda threads=256 blocks=64 accumulate_kernel!(total,out,lengths,Int32(32),Int32(64))
    end
end
logprob_study(factory)
