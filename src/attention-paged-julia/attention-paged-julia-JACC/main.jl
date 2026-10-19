using CUDA,Printf,JACC,StaticArrays
include("../device_functions.jl")
include("../study_common.jl")
@inline function uniform_kernel!(id,a,n,low,high)
 i=id-1
 if i<n;@inbounds a[i+1]=uniform_value(i,low,high);end
 return
end
@inline function normal_kernel!(id,a,n)
 i=id-1
 if i<n;@inbounds a[i+1]=normal_value(i);end
 return
end
@inline function attention_kernel!(id,out,q,k,v,table,lens,a,H,M,scale,::Val{D},::Val{N}) where {D,N}
 t=(id-1)%128;group=(id-1)÷128;h=group%H;s=group÷H
 sm=JACC.shared(SVector{N,Float32}(ntuple(_->0f0,Val(N))))
 load_query!(sm,q,t,h,s,H,Val(D));JACC.sync_workgroup()
 logits!(sm,k,table,lens,a,t,h,s,H,M,scale,Val(D));JACC.sync_workgroup()
 exponential!(sm,lens,t,s,Val(D));JACC.sync_workgroup()
 normalize!(sm,lens,t,s,Val(D));JACC.sync_workgroup()
 acc=values(sm,v,table,lens,t,h,s,H,M,Val(D));JACC.sync_workgroup()
 store_upper!(sm,acc,t,2,Val(D));JACC.sync_workgroup()
 acc=add_lower(sm,acc,t,2,Val(D));JACC.sync_workgroup()
 store_upper!(sm,acc,t,1,Val(D));JACC.sync_workgroup()
 acc=add_lower(sm,acc,t,1,Val(D));JACC.sync_workgroup()
 finish_output!(out,acc,t,h,s,H,Val(D))
 return
end
function uniform!(a,n,lo,hi)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,cld(n,256)*256,uniform_kernel!,a,n,lo,hi)
end
function normal!(a,n)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,256,normal_kernel!,a,n)
end
function attention!(out,q,k,v,table,lens,a,H,S,M,scale,D,N)
 spec=JACC.launch_spec(;threads=128,shmem_size=N*4,sync=false)
 JACC.parallel_for(spec,H*S*128,attention_kernel!,out,q,k,v,table,lens,a,Int32(H),Int32(M),scale,Val(D),Val(N))
end
paged_study(uniform!,normal!,attention!)
