include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
study((z,o,s)->(@cuda threads=64 blocks=(2,128) epi_kernel!(z,o,s,Int32(67),Int32(3),Int32(3),UInt32(7),UInt32(15))))
