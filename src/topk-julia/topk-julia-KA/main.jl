using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function topk_ka!(input,ids,values,batch::Int32,hidden::Int32,k::Int32,total::Int32)
    index=Int32(@index(Global,Linear))
    if index<=total
        zero=index-Int32(1);col=rem(zero,hidden);row=div(zero,hidden)
        value=@inbounds input[index];integer=Int32(value)
        if integer>=hidden-k && value==Float32(integer)
            rank=hidden-Int32(1)-integer;out=row*k+rank+Int32(1)
            @inbounds ids[out]=col
            @inbounds values[out]=value
        end
    end
end
function factory(input,ids,values)
    kernel=topk_ka!(get_backend(input),256)
    return ()->kernel(input,ids,values,Int32(128),Int32(4096),Int32(1024),Int32(524288);ndrange=524288)
end
topk_study(factory)
