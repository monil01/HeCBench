using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_tree_body.jl"))
@kernel function init_tree_ka!(pool,count,root)
 node=UInt32(0)
 if lane()==0;node=alloc!(count);end
 node=sh(node,Int32(0));@inbounds root[1]=node
 if lane()<2;putword!(pool,node,lane(),UInt32(1)-UInt32(lane()));end
end
@kernel function insert_tree_ka!(pool,count,root,keys,values,n)
 tid=Int32(@index(Global,Linear))-Int32(1)
 if tid-lane()<n
  key=UInt32(0);value=UInt32(0)
  if tid<n;key=@inbounds keys[tid+1]+UInt32(2);value=@inbounds values[tid+1]+UInt32(2);end
  insertunit!(pool,count,root,key,value,tid<n)
 end
end
@kernel function search_tree_ka!(pool,root,queries,results,n)
 tid=Int32(@index(Global,Linear))-Int32(1)
 if tid-lane()<n
  key=UInt32(0)
  if tid<n;key=@inbounds queries[tid+1]+UInt32(2);end
  value=searchunit(pool,root,key,tid<n)
  if tid<n&&value!=0;value-=UInt32(2);end
  @inbounds results[tid+1]=value
 end
end
backend=CUDA.CUDABackend()
init_kernel=init_tree_ka!(backend,32);insert_kernel=insert_tree_ka!(backend,256);search_kernel=search_tree_ka!(backend,256)
include(joinpath(@__DIR__,"..","study_common.jl"))
study((p,c,r)->init_kernel(p,c,r;ndrange=32),(p,c,r,k,v,n)->insert_kernel(p,c,r,k,v,n;ndrange=cld(n,256)*256),(p,r,q,out,n)->search_kernel(p,r,q,out,n;ndrange=cld(n,256)*256))
