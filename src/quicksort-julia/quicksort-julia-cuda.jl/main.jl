using CUDA
import Atomix
include(joinpath(@__DIR__,"..","study_device.jl"))
function qs_global_cuda!(d,dn,blocks,parents,news)
    shared=CuStaticSharedArray(UInt64,131)
    qs_global_body!(d,dn,blocks,parents,news,shared,Int32(blockIdx().x-1),Int32(threadIdx().x-1),CUDA.sync_threads)
    return
end
function qs_local_cuda!(d,dn,seqs)
    shared=CuStaticSharedArray(UInt64,(3712*sizeof(eltype(d))+1136)÷8)
    qs_local_body!(d,dn,seqs,shared,Int32(blockIdx().x-1),Int32(threadIdx().x-1),CUDA.sync_threads)
    return
end
allocate(v)=CuArray(v)
function launchglobal(c,count)
    @cuda threads=128 blocks=count qs_global_cuda!(c.d,c.dn,c.blocks,c.parents,c.news)
    nothing
end
function launchlocal(c,count)
    @cuda threads=128 blocks=count qs_local_cuda!(c.d,c.dn,c.done)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launchglobal,launchlocal)
