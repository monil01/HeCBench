using CUDA
import JACC, Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
function compaction_device(i,data,flags,n,tasks,cut,template)
 mem=JACC.shared(template)
 sc_body!(data,flags,n,tasks,cut,mem,Int32((i-1)÷256),Int32((i-1)%256),JACC.sync_workgroup)
 nothing
end
function launch(d,f,n,tasks,cut,template)
 spec=JACC.launch_spec(;threads=256,blocks=4,shmem_size=1028,sync=false)
 JACC.parallel_for(spec,1024,compaction_device,d,f,Int32(n),Int32(tasks),Int32(cut),template)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
