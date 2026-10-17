using CUDA
include(joinpath(@__DIR__,"..","study_lt.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==@__FILE__;mlp_study(mlp_lt_factory);end
