using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function bincount_global_jacc!(i,output,input)
    v=@inbounds input[i]
    if 0f0<=v<=1f0
        bin=get_bin(v,0f0,1f0,Int32(768))
        JACC.@atomic output[bin+1]+=Int32(1)
    end
end
function bincount_shared_jacc!(i,output,input,template)
    tid=mod(i-1,256)+1
    scratch=JACC.shared(template)
    v=@inbounds input[i]
    if 0f0<=v<=1f0
        bin=get_bin(v,0f0,1f0,Int32(768))
        JACC.@atomic scratch[bin+1]+=Int32(1)
    end
    JACC.sync_workgroup()
    for j in tid:256:768
        value=@inbounds scratch[j]
        JACC.@atomic output[j]+=value
    end
end
function factory(output,input,shared)
    spec=JACC.launch_spec(;threads=256,shmem_size=shared ? 3072 : 0,sync=false)
    if shared
        template=CUDA.zeros(Int32,768)
        return ()->JACC.parallel_for(spec,262144,bincount_shared_jacc!,output,input,template)
    else
        return ()->JACC.parallel_for(spec,262144,bincount_global_jacc!,output,input)
    end
end
bincount_study(factory)
