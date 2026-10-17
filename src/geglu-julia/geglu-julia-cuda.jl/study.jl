include(joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
println("STUDY_PACKAGE SpecialFunctions ",pkgversion(SpecialFunctions))
exit(study_main(study_prepare))
