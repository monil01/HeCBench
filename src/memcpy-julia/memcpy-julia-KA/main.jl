using CUDA,KernelAbstractions
const backend=CUDABackend()
transfer!(dest,source)=KernelAbstractions.copyto!(backend,dest,source)
include("../study_common.jl")
run_study(transfer!)
