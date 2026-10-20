using CUDA,KernelAbstractions,Libdl,SHA
include(joinpath(@__DIR__,"..","permutation_device.jl"))
@kernel function shuffle_generic_ka!(D,S,B,R,C,n,N,blocks,mean,median)
    tid=@index(Global,Linear)-1
    perm_shuffle!(S,B,D,n,N,tid,Val(false))
end
@kernel function shuffle_binary_ka!(D,S,B,R,C,n,N,blocks,mean,median)
    tid=@index(Global,Linear)-1
    perm_shuffle!(S,B,D,n,N,tid,Val(true))
end
@kernel function stats_generic_ka!(D,S,B,R,C,n,N,blocks,mean,median)
    index=@index(Global,Linear)-1
    perm_stats_generic!(S,B,R,C,n,N,rem(index,N),div(index,N),mean,median)
end
@kernel function stats_binary_ka!(D,S,B,R,C,n,N,blocks,mean,median)
    index=@index(Global,Linear)-1
    perm_stats_binary!(S,B,R,C,n,N,rem(index,N),div(index,N),mean,median)
end
const PERM_KA_BACKEND=CUDA.CUDABackend()
function perm_launch(q,D,S,B,R,C)
    args=(D,S,B,R,C,Int(q.len),Int(q.N),Int(q.blocks),q.mean,q.median)
    if q.size==1
        shuffle_binary_ka!(PERM_KA_BACKEND,Int(q.threads))(args...;ndrange=Int(q.N))
        stats_binary_ka!(PERM_KA_BACKEND,Int(q.threads))(args...;ndrange=4*Int(q.N))
    else
        shuffle_generic_ka!(PERM_KA_BACKEND,Int(q.threads))(args...;ndrange=Int(q.N))
        stats_generic_ka!(PERM_KA_BACKEND,Int(q.threads))(args...;ndrange=2*Int(q.N))
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
perm_study()
