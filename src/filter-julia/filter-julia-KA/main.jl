using CUDA
using KernelAbstractions
using Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_shared!(dst,count,src,scratch,n::Int32)
    idx=Int32(@index(Global,Linear));localidx=Int32(@index(Local,Linear));counter=@localmem Int32 (1,)
    if localidx==Int32(1)
        counter[1]=Int32(0)
    end
    @synchronize
    d=Int32(0);pos=Int32(0);keep=false
    if idx<=n
        d=@inbounds src[idx];keep=d>Int32(0)
        if keep
            pos=(Atomix.@atomic counter[1] + Int32(1))[1]
        end
    end
    @synchronize
    if localidx==Int32(1)
        counter[1]=(Atomix.@atomic count[1] + counter[1])[1]
    end
    @synchronize
    if keep
        @inbounds dst[counter[1]+pos+Int32(1)]=d
    end
    @synchronize
    nothing
end
@kernel function portable_global!(dst,count,src,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        d=@inbounds src[idx]
        if d>Int32(0)
            pos=(Atomix.@atomic count[1] + Int32(1))[1]
            @inbounds dst[pos+Int32(1)]=d
        end
    end
    nothing
end
function study_prepare(dst,count,src,n)
    scratch=CUDA.zeros(Int32,1);shared=portable_shared!(CUDABackend(),256);globalmap=portable_global!(CUDABackend(),256)
    function launch!(mode)
        if mode==0
            shared(dst,count,src,scratch,Int32(n);ndrange=n)
        else
            globalmap(dst,count,src,Int32(n);ndrange=n)
        end
    end
end
exit(study_main(study_prepare))
