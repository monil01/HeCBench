# No authored GPU kernels: genuine CUDA vendor-library host-controlled solver.
using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
bicg_study()
