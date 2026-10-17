using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function expand_ka!(starts,counts,edges,mask,updating,visited,cost,N)
 tid=@index(Global,Linear)
 @inbounds if tid<=N && mask[tid]!=Int8(0)
  mask[tid]=Int8(0);first=starts[tid];last=first+counts[tid];e=first
  while e<last
   id=edges[e+1]+Int32(1)
   if visited[id]==Int8(0);cost[id]=cost[tid]+Int32(1);updating[id]=Int8(1);end
   e+=Int32(1)
  end
 end
end
@kernel function update_ka!(mask,updating,visited,over,N)
 tid=@index(Global,Linear)
 @inbounds if tid<=N && updating[tid]!=Int8(0)
  mask[tid]=Int8(1);visited[tid]=Int8(1);over[1]=Int8(1);updating[tid]=Int8(0)
 end
end
function expand!(s,n,e,m,u,v,c,N)
 expand_ka!(CUDABackend(),256)(s,n,e,m,u,v,c,Int32(N);ndrange=cld(N,256)*256)
end
function update!(m,u,v,o,N)
 update_ka!(CUDABackend(),256)(m,u,v,o,Int32(N);ndrange=cld(N,256)*256)
end
study_main(expand!,update!)
