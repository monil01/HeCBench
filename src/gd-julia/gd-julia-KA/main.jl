using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function compute_ka!(x,grad,rows,cols,values,labels,obj,correct,m)
 i=Int32(@index(Global,Linear))-Int32(1)
 compute_body!(i,x,grad,rows,cols,values,labels,obj,correct,m)
end
@kernel cpu=false function update_ka!(x,grad,l2,m,n,lambda,alpha)
 tx=Int32(@index(Local,Linear));i=Int32(@index(Global,Linear))
 tile=@localmem Float32 (256,)
 @inbounds old=x[i];tile[tx]=old*old
 @synchronize
 for stride in (128,64,32,16,8,4,2,1)
  if tx<=stride;@inbounds tile[tx]+=tile[tx+stride];end
  @synchronize
 end
 if tx==1;CUDA.@atomic l2[1]+=tile[1];end
 @inbounds x[i]=old-alpha*(grad[i]/Float32(m)+lambda*old)
end
const BACKEND=CUDABackend()
function run!(x,grad,rows,cols,values,labels,obj,correct,l2,m,n,lambda,alpha,template)
 compute_ka!(BACKEND,256)(x,grad,rows,cols,values,labels,obj,correct,m;ndrange=512)
 update_ka!(BACKEND,256)(x,grad,l2,m,n,lambda,alpha;ndrange=512)
 nothing
end
study(run!)
