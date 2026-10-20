using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","permutation_device.jl"))
function shuffle_generic!(D,S,B,R,C,n,N,blocks,mean,median)
    tid=Int(threadIdx().x)-1+(Int(blockIdx().x)-1)*Int(blockDim().x)
    perm_shuffle!(S,B,D,n,N,tid,Val(false));nothing
end
function shuffle_binary!(D,S,B,R,C,n,N,blocks,mean,median)
    tid=Int(threadIdx().x)-1+(Int(blockIdx().x)-1)*Int(blockDim().x)
    perm_shuffle!(S,B,D,n,N,tid,Val(true));nothing
end
function stats_generic!(D,S,B,R,C,n,N,blocks,mean,median)
    block=Int(blockIdx().x)-1;tid=Int(threadIdx().x)-1+(block%blocks)*Int(blockDim().x)
    perm_stats_generic!(S,B,R,C,n,N,tid,div(block,blocks),mean,median);nothing
end
function stats_binary!(D,S,B,R,C,n,N,blocks,mean,median)
    block=Int(blockIdx().x)-1;tid=Int(threadIdx().x)-1+(block%blocks)*Int(blockDim().x)
    perm_stats_binary!(S,B,R,C,n,N,tid,div(block,blocks),mean,median);nothing
end
function perm_launch(q,D,S,B,R,C)
    args=(D,S,B,R,C,Int(q.len),Int(q.N),Int(q.blocks),q.mean,q.median)
    if q.size==1
        CUDA.@cuda threads=Int(q.threads) blocks=Int(q.blocks) stream=CUDA.legacy_stream() shuffle_binary!(args...)
        CUDA.@cuda threads=Int(q.threads) blocks=4*Int(q.blocks) stream=CUDA.legacy_stream() stats_binary!(args...)
    else
        CUDA.@cuda threads=Int(q.threads) blocks=Int(q.blocks) stream=CUDA.legacy_stream() shuffle_generic!(args...)
        CUDA.@cuda threads=Int(q.threads) blocks=2*Int(q.blocks) stream=CUDA.legacy_stream() stats_generic!(args...)
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
perm_study()
