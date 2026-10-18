using CUDA,JACC
JACC.@init_backend
transfer!(dest,source)=JACC.transfer!(dest,source)
include("../study_common.jl")
run_study(transfer!)
