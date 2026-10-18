using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);include_string(Main,first(split(source,"function main()")),joinpath(@__DIR__,"main.jl"))
include("../study_common.jl")
ao_study((img)->()->(@cuda threads=(16,16) blocks=(16,16) render_kernel!(img)))
