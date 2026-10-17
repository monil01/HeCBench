using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function reduce_squares!(a,partial,n)
    i=@index(Global,Linear);tid=@index(Local,Linear);block=@index(Group,Linear)
    scratch=@localmem Float32 (256,)
    @inbounds scratch[tid]=i<=n ? a[i]*a[i] : 0f0
    @synchronize
    offset=128
    while offset>=1
        if tid<=offset
            @inbounds scratch[tid]+=scratch[tid+offset]
        end
        @synchronize
        offset÷=2
    end
    if tid==1
        @inbounds partial[block]=scratch[1]
    end
end
@kernel function finish_norm!(partial,result,np)
    tid=@index(Local,Linear);scratch=@localmem Float32 (256,);s=0f0
    @inbounds for i in tid:256:np
        s+=partial[i]
    end
    @inbounds scratch[tid]=s
    @synchronize
    offset=128
    while offset>=1
        if tid<=offset
            @inbounds scratch[tid]+=scratch[tid+offset]
        end
        @synchronize
        offset÷=2
    end
    if tid==1
        @inbounds result[1]=sqrt(scratch[1])
    end
end
function study_prepare(a,n)
    np=cld(n,256);partial=CUDA.zeros(Float32,np);result=CUDA.zeros(Float32,1);reduce=reduce_squares!(CUDABackend(),256);finish=finish_norm!(CUDABackend(),256)
    function getnorm!()
        reduce(a,partial,Int32(n);ndrange=n);finish(partial,result,Int32(np);ndrange=256);only(Array(result))
    end
end
exit(study_main(study_prepare))
