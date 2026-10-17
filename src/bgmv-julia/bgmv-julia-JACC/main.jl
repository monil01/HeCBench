using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,output,input,weights,indices,template,::Val{BS},::Val{MODE}) where {BS,MODE}
 ii=Int32(i)-Int32(1);group=ii÷Int32(BS)
 tid=mod(ii,Int32(BS))+Int32(1);token=mod(group,Int32(TT));channel=group÷Int32(TT)
 tmp=JACC.shared(template)
 JACC.sync_workgroup()
 @inbounds tmp[tid]=partial(tid-Int32(1),token,channel,input,weights,indices,Val(BS),Val(MODE))
 JACC.sync_workgroup()
 stride=Int32(BS÷2)
 while stride>0
  if tid<=stride;@inbounds tmp[tid]+=tmp[tid+stride];end
  JACC.sync_workgroup();stride÷=Int32(2)
 end
 if tid==1;@inbounds store_result(output,token,channel,tmp[1],Val(MODE));end
 return
end
function run!(out,input,w,indices,tmp,::Val{BS},::Val{MODE}) where {BS,MODE}
 channels=MODE<=2 ? RR : HH
 JACC.parallel_for(JACC.launch_spec(threads=BS,blocks=TT*channels,shmem_size=4BS,sync=false),BS*TT*channels,kernel!,out,input,w,indices,tmp,Val(BS),Val(MODE))
 return
end
study(run!)
