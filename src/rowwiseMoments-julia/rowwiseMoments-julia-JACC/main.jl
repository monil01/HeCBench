using CUDA,JACC
JACC.@init_backend
include("../study_math.jl")
function rowwise_kernel(i,n,eps,x,mean,rstd)
 tid=Int32((i-1)%256)
 row=Int32((i-1)÷256+1)
 shared=CUDA.CuDynamicSharedArray(WData,32)
 moments_body(tid,row,n,eps,x,mean,rstd,shared)
 return
end
function launch(n,rows,eps,x,mean,rstd)
 specification=JACC.launch_spec(;threads=256,blocks=rows,shmem_size=768,sync=false)
 JACC.parallel_for(specification,rows*256,rowwise_kernel,n,eps,x,mean,rstd)
end
include("../study_common.jl")
run_study(launch)
