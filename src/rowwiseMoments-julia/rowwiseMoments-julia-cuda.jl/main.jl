using CUDA
include("../study_math.jl")
function rowwise_kernel(n,eps,x,mean,rstd)
 shared=CUDA.CuStaticSharedArray(WData,32)
 moments_body(Int32(threadIdx().x-1),Int32(blockIdx().x),n,eps,x,mean,rstd,shared)
 return
end
launch(n,rows,eps,x,mean,rstd)=(@cuda threads=256 blocks=rows rowwise_kernel(n,eps,x,mean,rstd))
include("../study_common.jl")
run_study(launch)
