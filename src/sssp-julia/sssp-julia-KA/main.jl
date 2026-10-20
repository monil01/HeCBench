# Derived from IMPACT SSSP, copyright 2016 University of Cordoba/Illinois.
# Full source redistribution terms are in ../LICENSE.source.
using CUDA,Libdl,KernelAbstractions,Atomix
@kernel function sssp_kernel!(@Const(nodes),@Const(edges),cost,color,@Const(qin),qout,sc)
 mem=@localmem Int32 (1603,)
 tid=(@index(Local,Linear))-1;gtid=(@index(Global,Linear))-1;wg=Int32(256)
 nt=@inbounds sc[1];shade=@inbounds sc[5]
 if tid==0
  @inbounds mem[1603]=0
  @inbounds mem[1602]=first(Atomix.@atomic :monotonic sc[2] + wg)
 end
 @synchronize();mybase=@inbounds mem[1602]
 while mybase<nt
  if (@inbounds mem[1603])>=800
   if tid==0;@inbounds mem[1601]=first(Atomix.@atomic :monotonic sc[3] + mem[1603]);end
   @synchronize();shift=tid
   while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
   @synchronize()
   if tid==0;@inbounds mem[1603]=0;end
   @synchronize()
  end
  @synchronize()
  if mybase+tid<nt && (@inbounds sc[4])==0
   pid=Int(@inbounds qin[Int(mybase)+tid+1]);(Atomix.@atomicswap :monotonic color[pid+1] = Int32(16677221));cur=@inbounds cost[pid+1];start=Int(@inbounds nodes[2pid+1]);degree=Int(@inbounds nodes[2pid+2])
   for j=start:start+degree-1
    id=Int(@inbounds edges[2j+1]);candidate=(@inbounds edges[2j+2])+cur;old=first(Atomix.@atomic :monotonic max(cost[id+1],candidate))
    if old<candidate
     oldcolor=first(Atomix.@atomic :monotonic max(color[id+1],shade))
     if oldcolor!=shade
      at=first(Atomix.@atomic :monotonic mem[1603] + Int32(1))
      if at>=1600;@inbounds sc[4]=1
      else;@inbounds mem[Int(at)+1]=Int32(id)
      end
     end
    end
   end
  end
  if tid==0;@inbounds mem[1602]=first(Atomix.@atomic :monotonic sc[2] + wg);end
  @synchronize();mybase=@inbounds mem[1602]
 end
 if tid==0;@inbounds mem[1601]=first(Atomix.@atomic :monotonic sc[3] + mem[1603]);end
 @synchronize();shift=tid
 while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
 if gtid==0;first(Atomix.@atomic :monotonic sc[6] + Int32(1));end
end
function launch!(nodes,edges,cost,color,qin,qout,sc,blocks)
 sssp_kernel!(KernelAbstractions.get_backend(nodes),256)(nodes,edges,cost,color,qin,qout,sc;ndrange=blocks*256)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main()
