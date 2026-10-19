using CUDA
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=CuArray(x)
function dot!(out,ref,q,n,blocks)
    tx=Int32(threadIdx().x);block=Int32(blockIdx().x)-1;tile=CuDynamicSharedArray(Int32,128)
    @inbounds tile[tx]=dot_partial(ref,q,tx,block,blocks,n)
    sync_threads();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        sync_threads();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
    return nothing
end
function count!(scores,threshold,out,n,blocks)
    tx=Int32(threadIdx().x);block=Int32(blockIdx().x)-1;tile=CuDynamicSharedArray(Int32,128)
    @inbounds tile[tx]=count_partial(scores,threshold,tx,block,blocks,n)
    sync_threads();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        sync_threads();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
    return nothing
end
function score!(random,ref,scores,m,uc,set,gens)
    idx=(Int32(blockIdx().x)-1)*Int32(128)+Int32(threadIdx().x)-1
    random_score!(idx,random,ref,scores,m,uc,set,gens)
    return nothing
end
launch_dot!(w)=@cuda threads=128 blocks=DB shmem=512 dot!(w.device.dot,w.device.ref,w.device.q,Int32(N),Int32(DB))
launch_scores!(w)=@cuda threads=128 blocks=SB shmem=0 score!(w.device.random,w.device.ref,w.device.scores,Int32(M),Float32(UC),Int32(w.set),Int32(R))
launch_count!(w,threshold)=@cuda threads=128 blocks=SB shmem=512 count!(w.device.scores,threshold,w.device.count,Int32(R),Int32(SB))
include("../study_common.jl")
study_main("cuda_jl")
