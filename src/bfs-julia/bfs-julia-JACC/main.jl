using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const BFS_BACKEND=JACC.get_backend(:cuda)
function bfs_expand_jacc!(tid,starts,counts,edges,mask,updating,visited,cost)
 @inbounds if mask[tid]!=Int8(0)
  mask[tid]=Int8(0);e=starts[tid];last=e+counts[tid]
  while e<last
   id=edges[e+1]+Int32(1)
   if visited[id]==Int8(0);cost[id]=cost[tid]+Int32(1);updating[id]=Int8(1);end
   e+=Int32(1)
  end
 end
 nothing
end
function bfs_update_jacc!(tid,mask,updating,visited,over)
 @inbounds if updating[tid]!=Int8(0)
  mask[tid]=Int8(1);visited[tid]=Int8(1);over[1]=Int8(1);updating[tid]=Int8(0)
 end
 nothing
end
function expand!(s,n,e,m,u,v,c,N)
 spec=JACC.LaunchSpec{typeof(BFS_BACKEND)}(;threads=256,blocks=cld(N,256),shmem_size=0,sync=false)
 JACC.parallel_for(bfs_expand_jacc!,spec,N,s,n,e,m,u,v,c)
end
function update!(m,u,v,o,N)
 spec=JACC.LaunchSpec{typeof(BFS_BACKEND)}(;threads=256,blocks=cld(N,256),shmem_size=0,sync=false)
 JACC.parallel_for(bfs_update_jacc!,spec,N,m,u,v,o)
end
study_main(expand!,update!)
