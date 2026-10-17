include(joinpath(@__DIR__,"gabor-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Named full Gabor study: 97x131, theta45 radians and0")
exit(study_main(study_prepare))
