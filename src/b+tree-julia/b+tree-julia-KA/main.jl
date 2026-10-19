using CUDA,KernelAbstractions
@kernel function point_kernel!(h,nodes,nn,records,curr,off,keys,ans)
 t=Int64(@index(Local,Linear))-1;b=@index(Group,Linear)
 @inbounds for level in Int64(1):h
  node=curr[b];key=keys[b]
  if nodes[node*517+258+t+1]<=key&&nodes[node*517+258+t+2]>key
   child=nodes[off[b]*517+t+2]
   if child<nn;off[b]=Int64(child);end
  end
  @synchronize()
  if t==0;curr[b]=off[b];end
  @synchronize()
 end
 @inbounds if nodes[curr[b]*517+258+t+1]==keys[b]
  ans[b]=records[Int64(nodes[curr[b]*517+t+2])+1]
 end
 nothing
end
@kernel function range_kernel!(h,nodes,nn,curr,off,last,off2,start,finish,rs,rl)
 t=Int64(@index(Local,Linear))-1;b=@index(Group,Linear)
 @inbounds for level in Int64(1):h
  node=curr[b];key=start[b]
  if nodes[node*517+258+t+1]<=key&&nodes[node*517+258+t+2]>key
   child=nodes[node*517+t+2]
   if child<nn;off[b]=Int64(child);end
  end
  node=last[b];key=finish[b]
  if nodes[node*517+258+t+1]<=key&&nodes[node*517+258+t+2]>key
   child=nodes[node*517+t+2]
   if child<nn;off2[b]=Int64(child);end
  end
  @synchronize()
  if t==0;curr[b]=off[b];last[b]=off2[b];end
  @synchronize()
 end
 @inbounds if nodes[curr[b]*517+258+t+1]==start[b];rs[b]=nodes[curr[b]*517+t+2];end
 @synchronize()
 @inbounds if nodes[last[b]*517+258+t+1]==finish[b];rl[b]=nodes[last[b]*517+t+2]-rs[b]+Int32(1);end
 nothing
end
const BACKEND=CUDABackend()
point(c)=point_kernel!(BACKEND,256)(Int64(c.h),c.nodes.data,Int64(c.nn),c.records.data,c.pc.data,c.po.data,c.keys.data,c.ans.data;ndrange=c.q*256)
rangequery(c)=range_kernel!(BACKEND,256)(Int64(c.h),c.nodes.data,Int64(c.nn),c.rc.data,c.ro.data,c.last.data,c.off2.data,c.start.data,c.finish.data,c.rs.data,c.rl.data;ndrange=c.q*256)
function alloc(v)
 a=KernelAbstractions.allocate(BACKEND,eltype(v),(length(v),));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,point,rangequery)
