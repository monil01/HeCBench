include(joinpath(@__DIR__,"bspline-vgh-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Complete64tapVGH,2003channels,4walkers,2coefficientfields,sourcebasisapproximationpreserved")
exit(study_main(study_prepare))
