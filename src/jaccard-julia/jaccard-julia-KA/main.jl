using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fill_kernel!(v,::Val{W}) where W
 j=@index(Global,Linear)
 if j<=length(v);@inbounds v[j]=W ? Float32(j)/Float32(length(v)) : 0f0;end
end
@kernel function row_kernel!(ptr,ind,v,work,mode)
 x,row=@index(Global,NTuple);row_body(x-Int32(1),row-Int32(1),ptr,ind,v,work,mode)
end
@kernel function inter_kernel!(ptr,ind,v,work,wi,ws,mode)
 x,y,row=@index(Global,NTuple);intersection_body(x-Int32(1),y-Int32(1),row-Int32(1),ptr,ind,v,work,wi,ws,mode)
end
@kernel function finish_kernel!(vals,wi,ws,wj)
 j=@index(Global,Linear);finish_body(j,vals,wi,ws,wj)
end
function run!(ptr,ind,vals,work,wi,ws,wj,mode)
 e=length(vals);be=get_backend(wj)
 fill_kernel!(be,256)(wj,mode;ndrange=cld(e,256)*256)
 fill_kernel!(be,256)(wi,Val(false);ndrange=cld(e,256)*256)
 row_kernel!(be,(16,4))(ptr,ind,wj,work,mode;ndrange=(16,NN))
 inter_kernel!(be,(8,4,8))(ptr,ind,wj,work,wi,ws,mode;ndrange=(8,4,NN))
 finish_kernel!(be,256)(vals,wi,ws,wj;ndrange=cld(e,256)*256)
 return
end
study(run!)
