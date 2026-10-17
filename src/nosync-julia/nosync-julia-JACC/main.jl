import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n))
