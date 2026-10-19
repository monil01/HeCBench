using CUDA,StaticArrays
include("../study_params.jl")
include("../study_hmm.jl")
allocate_model(x)=CuArray(x)
function mask_kernel!(seq,mat,table,p,n)
    gid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if gid<n;mask_sequence!(seq,mat,table,p,gid);end
    nothing
end
function observer_kernel!(seq,mat,p,post,errors,n)
    gid=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    if gid<n;observe_sequence!(seq,mat,p,post,errors,gid);end
    nothing
end
launch_mask!(w,mat,table,p)=(@cuda threads=128 blocks=(w.n+128)÷128 mask_kernel!(w.seq,mat,table,p,Int32(w.n)))
launch_observer!(w,mat,p,post,errors)=(@cuda threads=128 blocks=(w.n+128)÷128 observer_kernel!(w.seq,mat,p,post,errors,Int32(w.n)))
include("../study_common.jl")
study_main("cuda_jl")
