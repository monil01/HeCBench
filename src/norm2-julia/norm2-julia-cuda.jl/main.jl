using CUDA
function study_prepare(a,n)
    ()->CUDA.CUBLAS.nrm2(n,a)
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
