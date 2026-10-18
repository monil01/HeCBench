using CUDA
include("../study_math.jl")
launch(rhs,c,alpha,t)=(@cuda threads=96 blocks=1 rhs_kernel!(rhs,c,alpha,t))
include("../study_common.jl")
run_study(launch)
