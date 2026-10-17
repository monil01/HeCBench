# Named study entrypoint; archived historical CLI source and Makefile are preserved in the nested CUDA.jl directory.
include(joinpath(@__DIR__,"concat-julia-cuda.jl","main.jl"))
include(joinpath(@__DIR__,"study_common.jl"))
println("STUDY_CONFIGURATION allfourheads_seq32_batch1_beam2_headdim128_libcFixture_v1")
exit(study_main(study_prepare,ARGS))
