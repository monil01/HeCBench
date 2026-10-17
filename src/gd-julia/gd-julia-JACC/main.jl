using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function compute_jacc!(i,x,grad,rows,cols,values,labels,obj,correct,m)
 compute_body!(Int32(i)-Int32(1),x,grad,rows,cols,values,labels,obj,correct,m)
 nothing
end
@inline function update_jacc!(i,x,grad,l2,m,n,lambda,alpha,template)
 tx=mod(Int32(i)-Int32(1),Int32(256))+Int32(1)
 tile=JACC.shared(template)
 @inbounds old=x[i];tile[tx]=old*old
 JACC.sync_workgroup()
 for stride in (128,64,32,16,8,4,2,1)
  if tx<=stride;@inbounds tile[tx]+=tile[tx+stride];end
  JACC.sync_workgroup()
 end
 if tx==1;CUDA.@atomic l2[1]+=tile[1];end
 @inbounds x[i]=old-alpha*(grad[i]/Float32(m)+lambda*old)
 nothing
end
const COMPUTE_SPEC=JACC.launch_spec(threads=256,blocks=2,shmem_size=0,sync=false)
const UPDATE_SPEC=JACC.launch_spec(threads=256,blocks=2,shmem_size=1024,sync=false)
function run!(x,grad,rows,cols,values,labels,obj,correct,l2,m,n,lambda,alpha,template)
 JACC.parallel_for(COMPUTE_SPEC,512,compute_jacc!,x,grad,rows,cols,values,labels,obj,correct,m)
 JACC.parallel_for(UPDATE_SPEC,512,update_jacc!,x,grad,l2,m,n,lambda,alpha,template)
 nothing
end
study(run!)
