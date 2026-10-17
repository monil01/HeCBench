using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
study_allocate(A)=CuArray(A)
if abspath(PROGRAM_FILE)==@__FILE__;study_main(study_allocate);end
