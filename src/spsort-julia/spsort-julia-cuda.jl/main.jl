using CUDA
makewords(a)=CUDA.CuArray(a)
include("../study_common.jl")
spsort_study(makewords)
