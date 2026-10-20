# Derived from IMPACT SSSP, copyright 2016 University of Cordoba/Illinois.
# Full source redistribution terms are in ../LICENSE.source.
using CUDA,Libdl
function sssp_kernel!(nodes,edges,cost,color,qin,qout,sc)
 mem=CuStaticSharedArray(Int32,1603)
 tid=Int(threadIdx().x)-1;gtid=(Int(blockIdx().x)-1)*Int(blockDim().x)+tid;wg=Int32(blockDim().x)
 nt=@inbounds sc[1];shade=@inbounds sc[5]
 if tid==0
  @inbounds mem[1603]=0
  @inbounds mem[1602]=CUDA.atomic_add!(pointer(sc,2),wg)
 end
 CUDA.sync_threads();mybase=@inbounds mem[1602]
 while mybase<nt
  if (@inbounds mem[1603])>=800
   if tid==0;@inbounds mem[1601]=CUDA.atomic_add!(pointer(sc,3),mem[1603]);end
   CUDA.sync_threads();shift=tid
   while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
   CUDA.sync_threads()
   if tid==0;@inbounds mem[1603]=0;end
   CUDA.sync_threads()
  end
  CUDA.sync_threads()
  if mybase+tid<nt && (@inbounds sc[4])==0
   pid=Int(@inbounds qin[Int(mybase)+tid+1]);CUDA.atomic_xchg!(pointer(color,pid+1),Int32(16677221));cur=@inbounds cost[pid+1];start=Int(@inbounds nodes[2pid+1]);degree=Int(@inbounds nodes[2pid+2])
   for j=start:start+degree-1
    id=Int(@inbounds edges[2j+1]);candidate=(@inbounds edges[2j+2])+cur;old=CUDA.atomic_max!(pointer(cost,id+1),candidate)
    if old<candidate
     oldcolor=CUDA.atomic_max!(pointer(color,id+1),shade)
     if oldcolor!=shade
      at=CUDA.atomic_add!(pointer(mem,1603),Int32(1))
      if at>=1600;@inbounds sc[4]=1
      else;@inbounds mem[Int(at)+1]=Int32(id)
      end
     end
    end
   end
  end
  if tid==0;@inbounds mem[1602]=CUDA.atomic_add!(pointer(sc,2),wg);end
  CUDA.sync_threads();mybase=@inbounds mem[1602]
 end
 if tid==0;@inbounds mem[1601]=CUDA.atomic_add!(pointer(sc,3),mem[1603]);end
 CUDA.sync_threads();shift=tid
 while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
 if gtid==0;CUDA.atomic_add!(pointer(sc,6),Int32(1));end
 return
end
launch!(nodes,edges,cost,color,qin,qout,sc,blocks)=(@cuda threads=256 blocks=blocks sssp_kernel!(nodes,edges,cost,color,qin,qout,sc);nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main()
