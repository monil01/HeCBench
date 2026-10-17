using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function fpc_branch_kernel!(values,counter)
 tid=Int32(threadIdx().x);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+tid
 scratch=CuStaticSharedArray(UInt32,1)
 value=@inbounds values[i];inc=UInt32(fpc_increment(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 sync_threads()
 CUDA.@atomic scratch[1]+=inc
 sync_threads()
 if tid==Int32(256);CUDA.@atomic counter[1]+=scratch[1];end
return
end
function fpc_masks_kernel!(values,counter)
 tid=Int32(threadIdx().x);i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+tid
 scratch=CuStaticSharedArray(UInt32,1)
 value=@inbounds values[i];inc=UInt32(fpc_increment_masks(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 sync_threads()
 CUDA.@atomic scratch[1]+=inc
 sync_threads()
 if tid==Int32(256);CUDA.@atomic counter[1]+=scratch[1];end
return
end
function factory(values,counter,n,masked)
 kernel=masked ? fpc_masks_kernel! : fpc_branch_kernel!
 return ()->(@cuda threads=256 blocks=div(n,256) kernel(values,counter))
end
if abspath(PROGRAM_FILE)==@__FILE__;fpc_study(factory);end
