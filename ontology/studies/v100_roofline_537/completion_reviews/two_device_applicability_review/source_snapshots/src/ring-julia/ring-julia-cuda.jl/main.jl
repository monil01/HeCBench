include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA,Libdl
allocate(n)=CuArray{Int32}(undef,n)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate)
