# Derived from IMPACT SSSP, copyright 2016 University of Cordoba/Illinois.
# Full source redistribution terms are in ../LICENSE.source.
using CUDA,Libdl,JACC,Atomix
CUDA.stream!(CUDA.legacy_stream())
const SSSP_SEED_OWNER=CuArray(vcat(zeros(Int32,1603),fill(Int32(-19371337),17)))
const SSSP_SEED=view(SSSP_SEED_OWNER,1:1603)
const SSSP_JACC_SPECS=Dict(b=>JACC.launch_spec(threads=256,blocks=b,shmem_size=6412,sync=false,stream=CUDA.legacy_stream()) for b in (1,8,120))
function sssp_callback!(idx,nodes,edges,cost,color,qin,qout,sc,seed)
 mem=JACC.shared(seed)
 gtid=Int(idx)-1;tid=rem(gtid,256);wg=Int32(256)
 nt=@inbounds sc[1];shade=@inbounds sc[5]
 if tid==0
  @inbounds mem[1603]=0
  @inbounds mem[1602]=first(Atomix.@atomic :monotonic sc[2] + wg)
 end
 JACC.sync_workgroup();mybase=@inbounds mem[1602]
 while mybase<nt
  if (@inbounds mem[1603])>=800
   if tid==0;@inbounds mem[1601]=first(Atomix.@atomic :monotonic sc[3] + mem[1603]);end
   JACC.sync_workgroup();shift=tid
   while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
   JACC.sync_workgroup()
   if tid==0;@inbounds mem[1603]=0;end
   JACC.sync_workgroup()
  end
  JACC.sync_workgroup()
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
  JACC.sync_workgroup();mybase=@inbounds mem[1602]
 end
 if tid==0;@inbounds mem[1601]=first(Atomix.@atomic :monotonic sc[3] + mem[1603]);end
 JACC.sync_workgroup();shift=tid
 while shift<(@inbounds mem[1603]);@inbounds qout[Int(mem[1601])+shift+1]=mem[shift+1];shift+=Int(wg);end
 if gtid==0;first(Atomix.@atomic :monotonic sc[6] + Int32(1));end
end
function launch!(nodes,edges,cost,color,qin,qout,sc,blocks)
 JACC.parallel_for(SSSP_JACC_SPECS[blocks],blocks*256,sssp_callback!,nodes,edges,cost,color,qin,qout,sc,SSSP_SEED)
 nothing
end
function report_resource(phase)
 CUDA.device_synchronize(blocking=true);a=Array(SSSP_SEED_OWNER);@assert all(iszero,view(a,1:1603))&&all(==(Int32(-19371337)),view(a,1604:1620))
 print("STUDY_RESOURCE_INPUT {\"name\":\"JACC_shared_API_preload_seed_not_mathematical_graph_input\",\"phase\":\"",phase,"\",\"bytes\":6480,\"active_shared_bytes\":6412,\"timed_global_to_shared_preload\":true,\"int32_words\":[")
 for(k,x)in enumerate(a);print(k==1 ? "" : ",",x);end
 println("],\"all_seed_zero_and_tail_guards_exact\":true,\"host_input_readback_timed\":false}");flush(stdout)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
report_resource("before_all_trials")
study_main()
report_resource("after_all_trials")
