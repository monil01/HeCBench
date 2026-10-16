using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function nlll_ka!(output,total,input,target,weights)
    tid=@index(Local,Linear)
    scratch=@localmem Float32 (512,)
    acc=0f0;wacc=0f0
    for i in tid-1:256:4095
        t=@inbounds target[i+1]
        if t!=32
            w=@inbounds weights[t+1]
            acc-=(@inbounds input[i*64+t+1])*w;wacc+=w
        end
    end
    @inbounds scratch[tid]=acc
    @inbounds scratch[tid+256]=wacc
    @synchronize
    if tid==1
        loss=0f0;weight=0f0
        for j in 1:256
            loss+=@inbounds scratch[j];weight+=@inbounds scratch[j+256]
        end
        @inbounds output[1]=loss/weight
        @inbounds total[1]=weight
    end
end
function factory(out,total,input,target,weights)
    kernel=nlll_ka!(get_backend(input),256)
    return ()->kernel(out,total,input,target,weights;ndrange=256)
end
nlll_study(factory)
