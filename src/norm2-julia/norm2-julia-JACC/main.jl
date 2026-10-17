using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function reduce_squares!(i,a,partial,n,template)
    tid=(i-1)%256+1;block=(i-1)÷256+1;scratch=JACC.shared(template)
    @inbounds scratch[tid]=i<=n ? a[i]*a[i] : 0f0
    JACC.sync_workgroup();offset=128
    while offset>=1
        if tid<=offset
            @inbounds scratch[tid]+=scratch[tid+offset]
        end
        JACC.sync_workgroup();offset÷=2
    end
    if tid==1
        @inbounds partial[block]=scratch[1]
    end
    nothing
end
function finish_norm!(tid,partial,result,np,template)
    scratch=JACC.shared(template);s=0f0
    @inbounds for i in tid:256:np
        s+=partial[i]
    end
    @inbounds scratch[tid]=s
    JACC.sync_workgroup();offset=128
    while offset>=1
        if tid<=offset
            @inbounds scratch[tid]+=scratch[tid+offset]
        end
        JACC.sync_workgroup();offset÷=2
    end
    if tid==1
        @inbounds result[1]=sqrt(scratch[1])
    end
    nothing
end
function study_prepare(a,n)
    np=cld(n,256);partial=CUDA.zeros(Float32,np);result=CUDA.zeros(Float32,1);template=CUDA.zeros(Float32,256);backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};s1=ST(;threads=256,blocks=np,shmem_size=1024,sync=false);s2=ST(;threads=256,blocks=1,shmem_size=1024,sync=false)
    function getnorm!()
        JACC.parallel_for(reduce_squares!,s1,n,a,partial,Int32(n),template);JACC.parallel_for(finish_norm!,s2,256,partial,result,Int32(np),template);only(Array(result))
    end
end
exit(study_main(study_prepare))
