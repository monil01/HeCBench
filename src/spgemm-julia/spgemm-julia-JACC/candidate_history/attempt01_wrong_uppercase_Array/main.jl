using CUDA,JACC
JACC.@init_backend
makewords(a)=JACC.Array(a)
include("../study_common.jl")
spmm_study(makewords)
