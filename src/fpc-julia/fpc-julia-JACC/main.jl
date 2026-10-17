using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function fpc_branch_jacc!(index,values,counter,template)
 tid=Int32(mod(index-1,256)+1);i=Int32(index)
 scratch=JACC.shared(template)
 value=@inbounds values[i];inc=UInt32(fpc_increment(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 JACC.sync_workgroup()
 JACC.@atomic scratch[1]+=inc
 JACC.sync_workgroup()
 if tid==Int32(256);JACC.@atomic counter[1]+=scratch[1];end
end
function fpc_masks_jacc!(index,values,counter,template)
 tid=Int32(mod(index-1,256)+1);i=Int32(index)
 scratch=JACC.shared(template)
 value=@inbounds values[i];inc=UInt32(fpc_increment_masks(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 JACC.sync_workgroup()
 JACC.@atomic scratch[1]+=inc
 JACC.sync_workgroup()
 if tid==Int32(256);JACC.@atomic counter[1]+=scratch[1];end
end
function factory(values,counter,n,masked)
 kernel=masked ? fpc_masks_jacc! : fpc_branch_jacc!
 template=CUDA.zeros(UInt32,1)
 spec=JACC.launch_spec(;threads=256,shmem_size=4,sync=false)
 return ()->JACC.parallel_for(spec,n,kernel,values,counter,template)
end
fpc_study(factory)
