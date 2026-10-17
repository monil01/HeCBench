using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function fill_kernel!(v,::Val{W}) where W
 j=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if j<=length(v);@inbounds v[j]=W ? Float32(j)/Float32(length(v)) : 0f0;end
 return
end
function row_kernel!(ptr,ind,v,work,mode)
 row=(blockIdx().y-Int32(1))*Int32(4)+threadIdx().y-Int32(1);row_body(threadIdx().x-Int32(1),row,ptr,ind,v,work,mode);return
end
function inter_kernel!(ptr,ind,v,work,wi,ws,mode)
 row=(blockIdx().z-Int32(1))*Int32(8)+threadIdx().z-Int32(1);intersection_body(threadIdx().x-Int32(1),threadIdx().y-Int32(1),row,ptr,ind,v,work,wi,ws,mode);return
end
function finish_kernel!(vals,wi,ws,wj)
 j=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x;finish_body(j,vals,wi,ws,wj);return
end
function run!(ptr,ind,vals,work,wi,ws,wj,mode)
 e=length(vals);blocks=cld(e,256)
 @cuda threads=256 blocks=blocks fill_kernel!(wj,mode)
 @cuda threads=256 blocks=blocks fill_kernel!(wi,Val(false))
 @cuda threads=(16,4) blocks=(1,cld(NN,4)) row_kernel!(ptr,ind,wj,work,mode)
 @cuda threads=(8,4,8) blocks=(1,1,cld(NN,8)) inter_kernel!(ptr,ind,wj,work,wi,ws,mode)
 @cuda threads=256 blocks=blocks finish_kernel!(vals,wi,ws,wj)
 return
end
study(run!)
