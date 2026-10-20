using CUDA,JACC,Libdl,SHA
include(joinpath(@__DIR__,"..","permutation_device.jl"))
function shuffle_generic_jacc!(i,D,S,B,R,C,n,N,blocks,mean,median)
    perm_shuffle!(S,B,D,n,N,Int(i)-1,Val(false));nothing
end
function shuffle_binary_jacc!(i,D,S,B,R,C,n,N,blocks,mean,median)
    perm_shuffle!(S,B,D,n,N,Int(i)-1,Val(true));nothing
end
function stats_generic_jacc!(i,D,S,B,R,C,n,N,blocks,mean,median)
    index=Int(i)-1
    perm_stats_generic!(S,B,R,C,n,N,rem(index,N),div(index,N),mean,median);nothing
end
function stats_binary_jacc!(i,D,S,B,R,C,n,N,blocks,mean,median)
    index=Int(i)-1
    perm_stats_binary!(S,B,R,C,n,N,rem(index,N),div(index,N),mean,median);nothing
end
const PERM_JACC_SPECS=Dict{Tuple{Int,Int},Any}()
function perm_spec(threads,blocks)
    get!(PERM_JACC_SPECS,(threads,blocks)) do
        JACC.launch_spec(;threads=threads,blocks=blocks,shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
end
function perm_launch(q,D,S,B,R,C)
    args=(D,S,B,R,C,Int(q.len),Int(q.N),Int(q.blocks),q.mean,q.median)
    shuffle_spec=perm_spec(Int(q.threads),Int(q.blocks))
    if q.size==1
        stat_spec=perm_spec(Int(q.threads),4*Int(q.blocks))
        JACC.parallel_for(shuffle_spec,Int(q.N),shuffle_binary_jacc!,args...)
        JACC.parallel_for(stat_spec,4*Int(q.N),stats_binary_jacc!,args...)
    else
        stat_spec=perm_spec(Int(q.threads),2*Int(q.blocks))
        JACC.parallel_for(shuffle_spec,Int(q.N),shuffle_generic_jacc!,args...)
        JACC.parallel_for(stat_spec,2*Int(q.N),stats_generic_jacc!,args...)
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
perm_study()
