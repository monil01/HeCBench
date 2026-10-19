using CUDA,StaticArrays
import JACC
JACC.@init_backend
include("../study_params.jl")
include("../study_hmm.jl")
allocate_model(x)=JACC.array(x)
function mask_kernel!(i,seq,mat,table,p,n)
    gid=i-1
    if gid<n;mask_sequence!(seq,mat,table,p,gid);end
    nothing
end
function observer_kernel!(i,seq,mat,p,post,errors,n)
    gid=i-1
    if gid<n;observe_sequence!(seq,mat,p,post,errors,gid);end
    nothing
end
function launch_mask!(w,mat,table,p)
    blocks=(w.n+128)÷128
    JACC.parallel_for(JACC.launch_spec(threads=128,blocks=blocks,shmem_size=0,sync=false),128*blocks,mask_kernel!,w.seq,mat,table,p,Int32(w.n))
end
function launch_observer!(w,mat,p,post,errors)
    blocks=(w.n+128)÷128
    JACC.parallel_for(JACC.launch_spec(threads=128,blocks=blocks,shmem_size=0,sync=false),128*blocks,observer_kernel!,w.seq,mat,p,post,errors,Int32(w.n))
end
include("../study_common.jl")
study_main("jacc")
