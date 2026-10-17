include(joinpath(@__DIR__,"adv-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Named complete source-corner 3D cubature advection, N7/cubN15 E2, constantandlinear manufacturedfields")
exit(study_main(study_prepare))
