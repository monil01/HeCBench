using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function qs_ka_sync()
    @synchronize
    nothing
end
@kernel function qs_global_ka!(d,dn,blocks,parents,news)
    shared=@localmem UInt64 (131,)
    qs_global_body!(d,dn,blocks,parents,news,shared,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),qs_ka_sync)
end
@kernel function qs_local_ka!(d,dn,seqs,words::Val{W}) where {W}
    shared=@localmem UInt64 (W,)
    qs_local_body!(d,dn,seqs,shared,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),qs_ka_sync)
end
allocate(v)=begin
    a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),size(v));copyto!(a,v);a
end
function launchglobal(c,count)
    qs_global_ka!(KernelAbstractions.get_backend(c.d),128)(c.d,c.dn,c.blocks,c.parents,c.news;ndrange=count*128)
    nothing
end
function launchlocal(c,count)
    words=(3712*sizeof(eltype(c.d))+1136)÷8
    qs_local_ka!(KernelAbstractions.get_backend(c.d),128)(c.d,c.dn,c.done,Val(words);ndrange=count*128)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launchglobal,launchlocal)
