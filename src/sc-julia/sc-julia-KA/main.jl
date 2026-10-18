using CUDA, KernelAbstractions
import Atomix
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function sc_ka_sync()
 @synchronize
 nothing
end
@kernel function compaction_device(data,flags,n,tasks,cut)
 mem=@localmem Int32 (257,)
 sc_body!(data,flags,n,tasks,cut,mem,Int32(@index(Group,Linear))-Int32(1),Int32(@index(Local,Linear))-Int32(1),sc_ka_sync)
end
function launch(d,f,n,tasks,cut,template)
 compaction_device(KernelAbstractions.get_backend(d),256)(d,f,Int32(n),Int32(tasks),Int32(cut);ndrange=1024)
 nothing
end
function allocate(v)
 a=KernelAbstractions.allocate(CUDA.CUDABackend(),eltype(v),size(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
