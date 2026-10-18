using CUDA,KernelAbstractions
const H_MODEL=:ka
include(joinpath(@__DIR__,"..","hungarian_math.jl"))
@kernel unsafe_indices=true function hungarian_phase!(v,S,nt,nb)
 tid=Int32(@index(Local,Linear)-1);bid=Int32(@index(Group,Linear)-1)
 hphase!(v,tid,bid,nt,nb,S)
end
function launch!(v,S,nb,nt,shared)
 # Pinned KA CUDA call has no dynamic-shared-memory keyword. Explicit backend
 # adapter retains generated KA kernel/indexing and allocates required bytes.
 obj=hungarian_phase!(CUDABackend(),nt)
 nd,wg,it,dy=KernelAbstractions.launch_config(obj,nt*nb,nt)
 ctx=KernelAbstractions.mkcontext(obj,nd,it)
 CUDA.@cuda threads=nt blocks=nb shmem=shared maxthreads=nt obj.f(ctx,v,S,Int32(nt),Int32(nb))
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hungarian_study(launch!)
