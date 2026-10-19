using CUDA,KernelAbstractions,StaticArrays
include("../study_params.jl")
include("../study_hmm.jl")
allocate_model(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),length(x));copyto!(a,x);a
end
@kernel function mask_kernel!(seq,mat,table,p,n)
    gid=@index(Global,Linear)-1
    if gid<n;mask_sequence!(seq,mat,table,p,gid);end
end
@kernel function observer_kernel!(seq,mat,p,post,errors,n)
    gid=@index(Global,Linear)-1
    if gid<n;observe_sequence!(seq,mat,p,post,errors,gid);end
end
launch_mask!(w,mat,table,p)=mask_kernel!(CUDABackend(),128)(w.seq,mat,table,p,Int32(w.n);ndrange=128*((w.n+128)÷128))
launch_observer!(w,mat,p,post,errors)=observer_kernel!(CUDABackend(),128)(w.seq,mat,p,post,errors,Int32(w.n);ndrange=128*((w.n+128)÷128))
include("../study_common.jl")
study_main("ka")
