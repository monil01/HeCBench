using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function nlll_jacc!(tid,output,total,input,target,weights,workspace)
    acc=0f0;wacc=0f0
    for i in tid-1:256:4095
        t=@inbounds target[i+1]
        if t!=32
            w=@inbounds weights[t+1]
            acc-=(@inbounds input[i*64+t+1])*w;wacc+=w
        end
    end
    @inbounds workspace[tid]=acc
    @inbounds workspace[tid+256]=wacc
    JACC.sync_workgroup()
    scratch=JACC.shared(workspace)
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
    workspace=CUDA.zeros(Float32,512)
    spec=JACC.launch_spec(;threads=256,blocks=1,shmem_size=2048,sync=false)
    return ()->JACC.parallel_for(spec,256,nlll_jacc!,out,total,input,target,weights,workspace)
end
nlll_study(factory)
