using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function topk_jacc!(globalindex,input,ids,values,batch::Int32,hidden::Int32,k::Int32,total::Int32)
    index=Int32(globalindex)
    if index<=total
        zero=index-Int32(1);col=rem(zero,hidden);row=div(zero,hidden)
        value=@inbounds input[index];integer=Int32(value)
        if integer>=hidden-k && value==Float32(integer)
            rank=hidden-Int32(1)-integer;out=row*k+rank+Int32(1)
            @inbounds ids[out]=col
            @inbounds values[out]=value
        end
    end
    return
end
function factory(input,ids,values)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    return ()->JACC.parallel_for(spec,524288,topk_jacc!,input,ids,values,Int32(128),Int32(4096),Int32(1024),Int32(524288))
end
topk_study(factory)
