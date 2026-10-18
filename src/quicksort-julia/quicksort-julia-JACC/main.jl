using CUDA
import Atomix,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function qs_global_jacc!(index,d,dn,blocks,parents,news,template)
    shared=JACC.shared(template)
    qs_global_body!(d,dn,blocks,parents,news,shared,Int32((index-1)÷128),Int32((index-1)%128),JACC.sync_workgroup)
    nothing
end
function qs_local_jacc!(index,d,dn,seqs,template)
    shared=JACC.shared(template)
    qs_local_body!(d,dn,seqs,shared,Int32((index-1)÷128),Int32((index-1)%128),JACC.sync_workgroup)
    nothing
end
allocate(v)=JACC.array(v)
function launchglobal(c,count)
    spec=JACC.launch_spec(;threads=128,blocks=count,shmem_size=1048,sync=false)
    JACC.parallel_for(spec,count*128,qs_global_jacc!,c.d,c.dn,c.blocks,c.parents,c.news,c.global_template)
    nothing
end
function launchlocal(c,count)
    bytes=3712*sizeof(eltype(c.d))+1136
    spec=JACC.launch_spec(;threads=128,blocks=count,shmem_size=bytes,sync=false)
    JACC.parallel_for(spec,count*128,qs_local_jacc!,c.d,c.dn,c.done,c.local_template)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launchglobal,launchlocal)
