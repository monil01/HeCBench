include(joinpath(@__DIR__,"rainflow-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Complete257histories/rainflowcycles,counts+ranges+means,sourceendpoint/zerorangepolicy")
exit(study_main(study_prepare))
