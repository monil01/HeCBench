using CUDA,KernelAbstractions
include("../study_math.jl")
@kernel function rowwise_kernel(n,eps,x,mean,rstd)
 tid=@index(Local,Linear)
 row=@index(Group,Linear)
 shared=@localmem WData (32,)
 moments_body(Int32(tid-1),Int32(row),n,eps,x,mean,rstd,shared)
end
const object=rowwise_kernel(CUDABackend(),256)
launch(n,rows,eps,x,mean,rstd)=object(n,eps,x,mean,rstd;ndrange=256*rows)
include("../study_common.jl")
run_study(launch)
