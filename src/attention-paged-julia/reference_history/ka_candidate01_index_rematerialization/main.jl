using CUDA,Printf,KernelAbstractions
include("../device_functions.jl")
include("../study_common.jl")
@kernel function uniform_kernel!(a,n,low,high)
 i=@index(Global,Linear)
 i-=1
 if i<n;@inbounds a[i+1]=uniform_value(i,low,high);end
 nothing
end
@kernel function normal_kernel!(a,n)
 i=@index(Global,Linear)
 i-=1
 if i<n;@inbounds a[i+1]=normal_value(i);end
 nothing
end
@kernel function attention_kernel!(out,q,k,v,table,lens,a,H,M,scale,::Val{D},::Val{N}) where {D,N}
 t=@index(Local,Linear)
 group=@index(Group,Linear)
 t-=1;h=(group-1)%H;s=(group-1)÷H
 sm=@localmem Float32 (N,)
 load_query!(sm,q,t,h,s,H,Val(D));@synchronize()
 logits!(sm,k,table,lens,a,t,h,s,H,M,scale,Val(D));@synchronize()
 exponential!(sm,lens,t,s,Val(D));@synchronize()
 normalize!(sm,lens,t,s,Val(D));@synchronize()
 acc=values(sm,v,table,lens,t,h,s,H,M,Val(D));@synchronize()
 store_upper!(sm,acc,t,2,Val(D));@synchronize()
 acc=add_lower(sm,acc,t,2,Val(D));@synchronize()
 store_upper!(sm,acc,t,1,Val(D));@synchronize()
 acc=add_lower(sm,acc,t,1,Val(D));@synchronize()
 finish_output!(out,acc,t,h,s,H,Val(D))
 nothing
end
const backend=CUDA.CUDABackend()
uniform!(a,n,lo,hi)=uniform_kernel!(backend,256)(a,n,lo,hi;ndrange=cld(n,256)*256)
normal!(a,n)=normal_kernel!(backend,256)(a,n;ndrange=256)
function attention!(out,q,k,v,table,lens,a,H,S,M,scale,D,N)
 attention_kernel!(backend,128)(out,q,k,v,table,lens,a,Int32(H),Int32(M),scale,Val(D),Val(N);ndrange=128*H*S)
end
paged_study(uniform!,normal!,attention!)
