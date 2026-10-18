using CUDA
transfer!(dest,source)=copyto!(dest,source)
include("../study_common.jl")
run_study(transfer!)
