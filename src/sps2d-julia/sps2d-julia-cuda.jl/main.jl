using CUDA
makewords(a)=CUDA.CuArray(a)
include("../study_common.jl")
sps2d_study(makewords)
