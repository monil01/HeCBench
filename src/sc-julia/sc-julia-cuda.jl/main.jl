using CUDA
import Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function compaction_device(data,flags,n,tasks,cut)
 mem=CuStaticSharedArray(Int32,257)
 sc_body!(data,flags,n,tasks,cut,mem,Int32(blockIdx().x-1),Int32(threadIdx().x-1),CUDA.sync_threads)
 return
end
launch(d,f,n,tasks,cut,template)=(@cuda threads=256 blocks=4 compaction_device(d,f,Int32(n),Int32(tasks),Int32(cut));nothing)
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)
