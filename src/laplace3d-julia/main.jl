include(joinpath(@__DIR__,"laplace3d-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("Named20-step3DLaplace relaxation,64x19x17, boundary1andharmoniclinearfixtures")
exit(study_main(study_prepare))
