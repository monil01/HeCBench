include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA,Libdl
import JACC
JACC.@init_backend
allocate(n)=JACC.zeros(Int32,n)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate)
