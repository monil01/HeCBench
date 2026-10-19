using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","device_components.jl"))
@kernel unsafe_indices=true function sequence_ka!(inverse,a,b,oa,ob,ba,ea,bb,eb,scores,open,extend,scoring,encoding)
 bsw_device!(inverse,a,b,oa,ob,ba,ea,bb,eb,scores,open,extend,scoring,encoding)
end
function launch_ka!(inverse,threads,args...)
 # Pinned KA CUDA call exposes no dynamic shared-memory keyword; explicit
 # CUDA backend launch retains KA-generated kernel and exact source780bytes.
 obj=sequence_ka!(CUDABackend(),threads)
 nd,wg,it,dy=KernelAbstractions.launch_config(obj,N*threads,threads)
 ctx=KernelAbstractions.mkcontext(obj,nd,it)
 CUDA.@cuda threads=threads blocks=N shmem=SHMEM maxthreads=threads obj.f(ctx,inverse,args...)
 nothing
end
function allocate_ka(x)
 a=KernelAbstractions.allocate(CUDABackend(),eltype(x),(length(x),));copyto!(a,x);a
end
bsw_study(launch_ka!;allocate=allocate_ka)
