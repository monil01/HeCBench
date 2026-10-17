using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_tree_body.jl"))
function init_tree_jacc!(i,pool,count,root)
 node=UInt32(0)
 if lane()==0;node=alloc!(count);end
 node=sh(node,Int32(0));@inbounds root[1]=node
 if lane()<2;putword!(pool,node,lane(),UInt32(1)-UInt32(lane()));end
 return
end
function insert_tree_jacc!(i,pool,count,root,keys,values,n)
 tid=Int32(i)-Int32(1)
 if tid-lane()<n
  key=UInt32(0);value=UInt32(0)
  if tid<n;key=@inbounds keys[tid+1]+UInt32(2);value=@inbounds values[tid+1]+UInt32(2);end
  insertunit!(pool,count,root,key,value,tid<n)
 end
 return
end
function search_tree_jacc!(i,pool,root,queries,results,n)
 tid=Int32(i)-Int32(1)
 if tid-lane()<n
  key=UInt32(0)
  if tid<n;key=@inbounds queries[tid+1]+UInt32(2);end
  value=searchunit(pool,root,key,tid<n)
  if tid<n&&value!=0;value-=UInt32(2);end
  @inbounds results[tid+1]=value
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((p,c,r)->JACC.parallel_for(JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false),32,init_tree_jacc!,p,c,r),(p,c,r,k,v,n)->JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),cld(n,256)*256,insert_tree_jacc!,p,c,r,k,v,n),(p,r,q,out,n)->JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),cld(n,256)*256,search_tree_jacc!,p,r,q,out,n))
