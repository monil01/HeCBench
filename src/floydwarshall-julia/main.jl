include(joinpath(@__DIR__,"floydwarshall-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("CompleteN64APSP,2graphs,all64passes,distance+initializedintermediatematrices")
exit(study_main(study_prepare))
