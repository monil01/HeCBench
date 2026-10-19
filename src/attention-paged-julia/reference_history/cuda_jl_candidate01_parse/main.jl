using CUDA,Printf
include("../device_functions.jl")
include("../study_common.jl")
function uniform_kernel!(a,n,low,high)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 if i<n;@inbounds a[i+1]=uniform_value(i,low,high);end
 return
end
function normal_kernel!(a,n)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
 if i<n;@inbounds a[i+1]=normal_value(i);end
 return
end
function attention_kernel!(out,q,k,v,table,lens,a,H,M,scale,::Val{D},::Val{N}) where {D,N}
 t=Int(threadIdx().x)-1;h=Int(blockIdx().x)-1;s=Int(blockIdx().y)-1
 sm=CuStaticSharedArray(Float32,N)
 load_query!(sm,q,t,h,s,H,Val(D));sync_threads()
 logits!(sm,k,table,lens,a,t,h,s,H,M,scale,Val(D));sync_threads()
 exponential!(sm,lens,t,s,Val(D));sync_threads()
 normalize!(sm,lens,t,s,Val(D));sync_threads()
 acc=values(sm,v,table,lens,t,h,s,H,M,Val(D));sync_threads()
 store_upper!(sm,acc,t,2,Val(D));sync_threads()
 acc=add_lower(sm,acc,t,2,Val(D));sync_threads()
 store_upper!(sm,acc,t,1,Val(D));sync_threads()
 acc=add_lower(sm,acc,t,1,Val(D));sync_threads()
 finish_output!(out,acc,t,h,s,H,Val(D))
 return
end
uniform!(a,n,lo,hi)=(@cuda threads=256 blocks=cld(n,256) uniform_kernel!(a,n,lo,hi))
normal!(a,n)=(@cuda threads=256 blocks=cld(n,256) normal_kernel!(a,n))
function attention!(out,q,k,v,table,lens,a,H,S,M,scale,D,N)
 @cuda threads=128 blocks=(H,S) attention_kernel!(out,q,k,v,table,lens,a,Int32(H),Int32(M),scale,Val(D),Val(N))
end
paged_study(uniform!,normal!,attention!)
