using CUDA
include(joinpath(@__DIR__,"..","study_tree_body.jl"))
function init_tree!(pool,count,root)
 node=UInt32(0)
 if lane()==0;node=alloc!(count);end
 node=sh(node,Int32(0));@inbounds root[1]=node
 if lane()<2;putword!(pool,node,lane(),UInt32(1)-UInt32(lane()));end
 return
end
function insert_tree!(pool,count,root,keys,values,n)
 tid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 if tid-lane()>=n;return;end
 key=UInt32(0);value=UInt32(0)
 if tid<n;key=@inbounds keys[tid+1]+UInt32(2);value=@inbounds values[tid+1]+UInt32(2);end
 insertunit!(pool,count,root,key,value,tid<n);return
end
function search_tree!(pool,root,queries,results,n)
 tid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 if tid-lane()>=n;return;end
 key=UInt32(0)
 if tid<n;key=@inbounds queries[tid+1]+UInt32(2);end
 value=searchunit(pool,root,key,tid<n)
 if tid<n&&value!=0;value-=UInt32(2);end
 @inbounds results[tid+1]=value;return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((p,c,r)->(@cuda threads=32 init_tree!(p,c,r)),(p,c,r,k,v,n)->(@cuda threads=256 blocks=cld(n,256) insert_tree!(p,c,r,k,v,n)),(p,r,q,out,n)->(@cuda threads=256 blocks=cld(n,256) search_tree!(p,r,q,out,n)))
