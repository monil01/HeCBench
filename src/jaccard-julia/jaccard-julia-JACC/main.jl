using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function fill_kernel!(j,v,::Val{W}) where W
 if j<=length(v);@inbounds v[j]=W ? Float32(j)/Float32(length(v)) : 0f0;end
 return
end
row_kernel!(x,row,ptr,ind,v,work,mode)=row_body(x-Int32(1),row-Int32(1),ptr,ind,v,work,mode)
inter_kernel!(x,y,row,ptr,ind,v,work,wi,ws,mode)=intersection_body(x-Int32(1),y-Int32(1),row-Int32(1),ptr,ind,v,work,wi,ws,mode)
finish_kernel!(j,vals,wi,ws,wj)=finish_body(j,vals,wi,ws,wj)
function run!(ptr,ind,vals,work,wi,ws,wj,mode)
 e=length(vals);blocks=cld(e,256);s=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(s,blocks*256,fill_kernel!,wj,mode)
 JACC.parallel_for(s,blocks*256,fill_kernel!,wi,Val(false))
 JACC.parallel_for(JACC.launch_spec(threads=(16,4),blocks=(1,cld(NN,4)),shmem_size=0,sync=false),(16,NN),row_kernel!,ptr,ind,wj,work,mode)
 JACC.parallel_for(JACC.launch_spec(threads=(8,4,8),blocks=(1,1,cld(NN,8)),shmem_size=0,sync=false),(8,4,NN),inter_kernel!,ptr,ind,wj,work,wi,ws,mode)
 JACC.parallel_for(s,blocks*256,finish_kernel!,vals,wi,ws,wj)
 return
end
study(run!)
