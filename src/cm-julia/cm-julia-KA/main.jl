using CUDA, KernelAbstractions
include("../study_params.jl")
include("../device_common.jl")
allocate_model(x)=begin
 a=KernelAbstractions.allocate(CUDABackend(),eltype(x),size(x)...);copyto!(a,x);a
end
@kernel function dot!(out,ref,q,n,blocks)
    tx=Int32(@index(Local,Linear));block=Int32(@index(Group,Linear))-1;tile=@localmem Int32 (128,)
    @inbounds tile[tx]=dot_partial(ref,q,tx,block,blocks,n)
    @synchronize();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        @synchronize();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
end
@kernel function cm_count!(scores,threshold,out,n,blocks)
    tx=Int32(@index(Local,Linear));block=Int32(@index(Group,Linear))-1;tile=@localmem Int32 (128,)
    @inbounds tile[tx]=count_partial(scores,threshold,tx,block,blocks,n)
    @synchronize();stride=Int32(64)
    while stride!=0
        if tx<=stride
            @inbounds tile[tx]+=tile[tx+stride]
        end
        @synchronize();stride÷=2
    end
    if tx==1;@inbounds out[block+17]=tile[1];end
end
@kernel function score!(random,ref,scores,m,uc,set,gens)
    idx=Int32(@index(Global,Linear))-1
    random_score!(idx,random,ref,scores,m,uc,set,gens)
end
launch_dot!(w)=dot!(CUDABackend(),128)(w.device.dot,w.device.ref,w.device.q,Int32(N),Int32(DB);ndrange=DB*128)
launch_scores!(w)=score!(CUDABackend(),128)(w.device.random,w.device.ref,w.device.scores,Int32(M),Float32(UC),Int32(w.set),Int32(R);ndrange=SB*128)
launch_count!(w,threshold)=cm_count!(CUDABackend(),128)(w.device.scores,threshold,w.device.count,Int32(R),Int32(SB);ndrange=SB*128)
include("../study_common.jl")
study_main("ka")
