using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function bincount_global_ka!(output,input)
    i=@index(Global,Linear)
    if i<=262144
        v=@inbounds input[i]
        if 0f0<=v<=1f0
            bin=get_bin(v,0f0,1f0,Int32(768))
            Atomix.@atomic output[bin+1]+=Int32(1)
        end
    end
end
@kernel function bincount_shared_ka!(output,input)
    tid=@index(Local,Linear);i=@index(Global,Linear)
    scratch=@localmem Int32 (768,)
    for j in tid:256:768;@inbounds scratch[j]=Int32(0);end
    @synchronize
    if i<=262144
        v=@inbounds input[i]
        if 0f0<=v<=1f0
            bin=get_bin(v,0f0,1f0,Int32(768))
            Atomix.@atomic scratch[bin+1]+=Int32(1)
        end
    end
    @synchronize
    for j in tid:256:768
        value=@inbounds scratch[j]
        Atomix.@atomic output[j]+=value
    end
end
function factory(output,input,shared)
    kernel=shared ? bincount_shared_ka!(get_backend(input),256) : bincount_global_ka!(get_backend(input),256)
    return ()->kernel(output,input;ndrange=262144)
end
bincount_study(factory)
