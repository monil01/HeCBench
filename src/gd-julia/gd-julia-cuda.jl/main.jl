include(joinpath(@__DIR__,"..","study_common.jl"))
function compute_ref!(x,grad,rows,cols,values,labels,obj,correct,m)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 compute_body!(i,x,grad,rows,cols,values,labels,obj,correct,m)
 return
end
function update_ref!(x,grad,l2,m,n,lambda,alpha)
 tx=threadIdx().x;i=(blockIdx().x-Int32(1))*blockDim().x+tx
 tile=CuStaticSharedArray(Float32,256)
 @inbounds old=x[i];tile[tx]=old*old
 sync_threads()
 stride=Int32(128)
 while stride>0
  if tx<=stride;@inbounds tile[tx]+=tile[tx+stride];end
  sync_threads();stride>>=1
 end
 if tx==1;CUDA.@atomic l2[1]+=tile[1];end
 @inbounds x[i]=old-alpha*(grad[i]/Float32(m)+lambda*old)
 return
end
function run!(x,grad,rows,cols,values,labels,obj,correct,l2,m,n,lambda,alpha,template)
 @cuda threads=256 blocks=2 compute_ref!(x,grad,rows,cols,values,labels,obj,correct,m)
 @cuda threads=256 blocks=2 update_ref!(x,grad,l2,m,n,lambda,alpha)
 nothing
end
study(run!)
