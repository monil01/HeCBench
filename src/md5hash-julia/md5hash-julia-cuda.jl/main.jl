using CUDA
include(joinpath(@__DIR__,"..","md5_math.jl"))
function search_kernel!(fi,fk,fd,n,len,base,s0,s1,s2,s3)
 tid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
 study_search_thread!(tid,fi,fk,fd,n,len,base,s0,s1,s2,s3)
 return
end
function probe_kernel!(w0,w1,lens,out,n)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 study_probe_thread!(i,w0,w1,lens,out,n)
 return
end
