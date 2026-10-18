using CUDA
makewords(a)=CUDA.CuArray(a)
include("../study_common.jl")
spmm_study(makewords)
