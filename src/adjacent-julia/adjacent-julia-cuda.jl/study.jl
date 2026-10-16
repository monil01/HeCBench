using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,first(split(source,"function initialize")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(input,output)
    return ()->begin
        @cuda threads=256 blocks=1024 block_adj_diff_kernel!(input,output,Int32(1),Int32(256))
        @cuda threads=256 blocks=1024 block_adj_diff_kernel!(output,output,Int32(0),Int32(256))
    end
end
adjacent_study(factory)
